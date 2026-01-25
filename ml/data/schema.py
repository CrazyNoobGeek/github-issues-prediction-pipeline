from typing import TypedDict, List, Optional


class IssueRecord(TypedDict):
    repo_full_name: str
    number: int

    title: Optional[str]
    body: Optional[str]
    labels: List[str]
    author_association: Optional[str]

    comments: int
    num_assignees: int

    created_at: str
    updated_at: Optional[str]
    closed_at: Optional[str]
    first_comment_at: Optional[str]

    is_pull_request: bool