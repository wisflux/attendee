from django.core.management.base import BaseCommand

from bots.bots_api_utils import retry_failed_transcription
from bots.models import Bot


class Command(BaseCommand):
    help = "Re-enqueues transcription for a bot's failed utterances (audio is retained on failure)."

    def add_arguments(self, parser):
        parser.add_argument("--bot-object-id", type=str, required=True, help="The bot's object id, e.g. bot_xxxxxxxxxxx")

    def handle(self, *args, **options):
        bot = Bot.objects.get(object_id=options["bot_object_id"])
        retried_count, error = retry_failed_transcription(bot)
        if error:
            self.stderr.write(f"Error: {error['error']}")
            return
        self.stdout.write(f"Re-enqueued transcription for {retried_count} failed utterances of bot {bot.object_id}")
