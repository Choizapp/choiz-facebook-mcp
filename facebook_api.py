import requests
from datetime import datetime, timezone
from typing import Any
from config import GRAPH_API_BASE_URL, PAGE_ID, PAGE_ACCESS_TOKEN

# Keys whose values are fully-formed Graph API URLs with the token in the query
# string. Graph puts them under every "paging" object it returns.
_PAGING_URL_KEYS = ("next", "previous")


def _strip_paging_urls(node: Any) -> Any:
    """Recursively drop Graph API paging URLs, which embed the Page token.

    Graph returns ``paging.next``/``paging.previous`` as ready-to-call URLs with
    ``access_token=<PAGE_ACCESS_TOKEN>`` in the query string. Whatever _request
    returns goes straight back to the MCP client, so handing those URLs over
    verbatim leaks the Page access token into a third-party chat transcript on
    every call to a paginated edge. Today that is get_page_posts,
    get_post_comments and get_post_insights, plus anything paginated added
    later -- which is why this is enforced here in the transport rather than
    per tool.

    The opaque cursors under ``paging.cursors`` carry no credentials, so they
    are kept: they are what a caller needs to page forward, by passing the
    cursor back as an ``after`` parameter. ``has_next`` replaces the stripped
    ``next`` URL so callers can still tell whether more data exists.

    Nested edges (comments inside posts, attachments inside comments) get their
    own paging blocks, hence the recursion.
    """
    if isinstance(node, list):
        return [_strip_paging_urls(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned = {key: _strip_paging_urls(value) for key, value in node.items()}

    paging = cleaned.get("paging")
    if isinstance(paging, dict):
        safe = {k: v for k, v in paging.items() if k not in _PAGING_URL_KEYS}
        if paging.get("next"):
            safe["has_next"] = True
        cleaned["paging"] = safe

    return cleaned


# Hard ceiling on how many posts a single call may return. Trimmed posts run
# ~300 bytes each, so 200 is roughly 60 KB -- past that a single tool result
# gets unwieldy for a client to consume in one turn.
MAX_POSTS = 200

# Graph caps `limit` on the posts edge at 100 per request.
_GRAPH_PAGE_SIZE = 100

# Bound on round-trips per call. Normally MAX_POSTS/_GRAPH_PAGE_SIZE = 2 pages,
# but a since/until window can yield sparse pages, so allow slack while still
# refusing to loop forever.
_MAX_PAGES = 10

# full_picture is deliberately absent: each is a ~750-character signed CDN URL
# that more than doubles the payload and expires anyway. It is opt-in via
# include_images. permalink_url stays -- short, stable, and what a human needs
# to actually open the post.
_POST_FIELDS = (
    "id",
    "message",
    "created_time",
    "permalink_url",
    "shares",
    "attachments{type,media_type,title,description}",
)


def _parse_time_bound(value: Any) -> Any:
    """Best-effort parse of a since/until bound into an aware datetime.

    Accepts what Graph accepts -- a unix timestamp or a date string -- and
    returns None when it cannot tell, in which case the local re-filter simply
    does not apply that bound.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) or (
        isinstance(value, str) and value.strip().lstrip("-").isdigit()
    ):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _within_window(post: Any, since_dt: Any, until_dt: Any) -> bool:
    """Whether a post falls inside [since_dt, until_dt].

    Anything unparseable is kept rather than silently dropped: a missing or odd
    created_time is not grounds for hiding a post from the caller.
    """
    if since_dt is None and until_dt is None:
        return True
    if not isinstance(post, dict):
        return True
    raw = post.get("created_time")
    if not raw:
        return True
    try:
        created = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return True
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if since_dt is not None and created < since_dt:
        return False
    if until_dt is not None and created > until_dt:
        return False
    return True


class FacebookAPI:
    # Generic Graph API request method
    def _request(self, method: str, endpoint: str, params: dict[str, Any], json: dict[str, Any] = None) -> dict[str, Any]:
        url = f"{GRAPH_API_BASE_URL}/{endpoint}"
        params["access_token"] = PAGE_ACCESS_TOKEN
        response = requests.request(method, url, params=params, json=json)
        # Never hand Graph's paging URLs to the caller: they carry the token.
        return _strip_paging_urls(response.json())

    def post_message(self, message: str) -> dict[str, Any]:
        return self._request("POST", f"{PAGE_ID}/feed", {"message": message})

    def reply_to_comment(self, comment_id: str, message: str) -> dict[str, Any]:
        return self._request("POST", f"{comment_id}/comments", {"message": message})

    def get_posts(
        self,
        limit: int = 25,
        since: Any = None,
        until: Any = None,
        include_images: bool = False,
    ) -> dict[str, Any]:
        """Fetch Page posts, walking Graph's cursors until `limit` is met.

        Graph defaults the posts edge to 25 per page -- that default is where
        the old "only the last 25 posts" behaviour came from. This paginates
        instead of exposing a raw cursor to the MCP client, which would mean
        threading an 800-character opaque string back and forth between turns.

        Pagination follows `paging.cursors.after`, not `paging.next`, because
        _strip_paging_urls removes the next URL (it embeds the access token).
        The two are equivalent: the URL Graph builds for `next` is this same
        request plus `&after=<cursor>`.

        since/until are passed through to Graph, which documents time-based
        pagination but does not enumerate which edges honour it. The window is
        therefore re-applied locally on the way out, so the result respects the
        caller's dates whether or not the edge filtered server-side.
        """
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 25
        limit = max(1, min(limit, MAX_POSTS))

        fields = list(_POST_FIELDS)
        if include_images:
            fields.append("full_picture")

        base_params: dict[str, Any] = {
            "fields": ",".join(fields),
            "limit": min(limit, _GRAPH_PAGE_SIZE),
        }
        if since:
            base_params["since"] = since
        if until:
            base_params["until"] = until

        since_dt = _parse_time_bound(since)
        until_dt = _parse_time_bound(until)

        posts: list[Any] = []
        after: Any = None
        pages = 0
        exhausted = False

        while len(posts) < limit and pages < _MAX_PAGES:
            params = dict(base_params)
            if after:
                params["after"] = after

            payload = self._request("GET", f"{PAGE_ID}/posts", params)
            pages += 1

            # Surface Graph errors (bad token, invalid window) untouched rather
            # than returning a confusingly empty list.
            if isinstance(payload, dict) and payload.get("error"):
                return payload
            if not isinstance(payload, dict):
                break

            posts.extend(
                post
                for post in (payload.get("data") or [])
                if _within_window(post, since_dt, until_dt)
            )

            paging = payload.get("paging") or {}
            if not paging.get("has_next"):
                exhausted = True
                break
            after = (paging.get("cursors") or {}).get("after")
            if not after:
                exhausted = True
                break

        returned = posts[:limit]
        result: dict[str, Any] = {
            "data": returned,
            "count": len(returned),
            "pages_fetched": pages,
            # True only when we ran out of posts rather than hitting `limit`, so
            # a caller can tell "that is all there is" from "there may be more".
            "reached_oldest_post": exhausted and len(posts) <= limit,
        }
        if not include_images:
            result["note"] = (
                "Image URLs omitted to keep the response small; pass "
                "include_images=true if you need full_picture."
            )
        return result

    def get_comments(self, post_id: str) -> dict[str, Any]:
        return self._request("GET", f"{post_id}/comments", {"fields": "id,message,from,created_time"})

    def delete_post(self, post_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"{post_id}", {})

    def delete_comment(self, comment_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"{comment_id}", {})

    def hide_comment(self, comment_id: str) -> dict[str, Any]:
        """Hide a comment from the Page."""
        return self._request("POST", f"{comment_id}", {"is_hidden": True})

    def unhide_comment(self, comment_id: str) -> dict[str, Any]:
        """Unhide a previously hidden comment."""
        return self._request("POST", f"{comment_id}", {"is_hidden": False})

    def get_insights(self, post_id: str, metric: str, period: str = "lifetime") -> dict[str, Any]:
        return self._request("GET", f"{post_id}/insights", {"metric": metric, "period": period})

    def get_bulk_insights(self, post_id: str, metrics: list[str], period: str = "lifetime") -> dict[str, Any]:
        metric_str = ",".join(metrics)
        return self.get_insights(post_id, metric_str, period)

    def post_image_to_facebook(self, image_url: str, caption: str) -> dict[str, Any]:
        params = {
            "url": image_url,
            "caption": caption
        }
        return self._request("POST", f"{PAGE_ID}/photos", params)
    
    def send_dm_to_user(self, user_id: str, message: str) -> dict[str, Any]:
        payload = {
            "recipient": {"id": user_id},
            "message": {"text": message},
            "messaging_type": "RESPONSE"
        }
        return self._request("POST", "me/messages", {}, json=payload)
    
    def update_post(self, post_id: str, new_message: str) -> dict[str, Any]:
        return self._request("POST", f"{post_id}", {"message": new_message})

    def schedule_post(self, message: str, publish_time: int) -> dict[str, Any]:
        params = {
            "message": message,
            "published": False,
            "scheduled_publish_time": publish_time,
        }
        return self._request("POST", f"{PAGE_ID}/feed", params)

    def get_page_fan_count(self) -> int:
        data = self._request("GET", f"{PAGE_ID}", {"fields": "fan_count"})
        return data.get("fan_count", 0)

    def get_post_share_count(self, post_id: str) -> int:
        data = self._request("GET", f"{post_id}", {"fields": "shares"})
        return data.get("shares", {}).get("count", 0)
