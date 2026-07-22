from django.conf import settings
from rest_framework.throttling import SimpleRateThrottle


class ProjectRateThrottle(SimpleRateThrottle):
    """
    Throttle by project (from request.auth.project)
    """

    scope = "project"

    def get_cache_key(self, request, view):
        auth = getattr(request, "auth", None)
        project = getattr(auth, "project", None)
        if not project:
            return None

        ident = getattr(project, "object_id", "unknown")
        return self.cache_format % {"scope": self.scope, "ident": ident}


class ProjectPostThrottle(ProjectRateThrottle):
    scope = "project_post"

    def allow_request(self, request, view):
        if request.method != "POST":
            return True
        if settings.DISABLE_RATE_LIMITING:
            return True
        return super().allow_request(request, view)


class MemberReadThrottle(SimpleRateThrottle):
    """Rate-limit reads by the MEMBER making them, not by their project.

    Every desktop install shares one project API key, so throttling per project would let a
    single runaway client exhaust the budget for everyone. Keying on the team.day user id keeps
    a misbehaving client limited to itself. Requests we cannot attribute to a member are left
    alone -- they are rejected by the view's auth anyway.
    """

    scope = "member_read"

    def allow_request(self, request, view):
        if request.method not in ("GET", "HEAD"):
            return True
        if settings.DISABLE_RATE_LIMITING:
            return True
        return super().allow_request(request, view)

    def get_cache_key(self, request, view):
        from bots.team_day_user_auth import quiet_user_id

        user_id = quiet_user_id(request)
        if not user_id:
            return None
        return self.cache_format % {"scope": self.scope, "ident": user_id}
