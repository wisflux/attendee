from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("bots", "0094_meetingsharetoken"),
    ]

    operations = [
        migrations.AlterField(
            model_name="credentials",
            name="credential_type",
            field=models.IntegerField(choices=[(1, "Deepgram"), (2, "Zoom OAuth"), (3, "Google Text To Speech"), (4, "Gladia"), (5, "OpenAI"), (6, "Assembly AI"), (7, "Sarvam"), (8, "Teams Bot Login"), (9, "External Media Storage"), (10, "ElevenLabs"), (11, "Kyutai"), (12, "Azure OpenAI")]),
        ),
    ]
