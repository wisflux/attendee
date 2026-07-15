from django.contrib import admin

from .models import AppUser, LiveUtterance, Meeting, MeetingSummary

admin.site.register(AppUser)
admin.site.register(Meeting)
admin.site.register(LiveUtterance)
admin.site.register(MeetingSummary)
