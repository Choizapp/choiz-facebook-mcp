import requests
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

    def get_posts(self) -> dict[str, Any]:
        return self._request("GET", f"{PAGE_ID}/posts", {
            "fields": "id,message,created_time,permalink_url,shares,full_picture,attachments{type,media_type,title,description}"
        })

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
