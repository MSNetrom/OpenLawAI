import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("chatdb", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="UsageRequestSummary",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("request_label", models.CharField(max_length=50)),
                ("input_tokens", models.PositiveIntegerField(default=0)),
                ("output_tokens", models.PositiveIntegerField(default=0)),
                ("search_performed", models.BooleanField(default=False)),
                ("ocr_performed", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name="usage_summaries", to=settings.AUTH_USER_MODEL)),
                ("conversation", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="usage_summaries", to="chatdb.chatconversation")),
            ],
            options={
                "ordering": ["-created_at"],
                "indexes": [
                    models.Index(fields=["user", "created_at"], name="usage_usager_user_id_d3e4f5_idx"),
                ],
            },
        ),
        migrations.CreateModel(
            name="UsageCallLedger",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("model", models.CharField(max_length=64)),
                ("input_tokens", models.PositiveIntegerField()),
                ("output_tokens", models.PositiveIntegerField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("request_summary", models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name="calls", to="usage.usagerequestsummary")),
            ],
            options={
                "ordering": ["created_at"],
            },
        ),
    ]
