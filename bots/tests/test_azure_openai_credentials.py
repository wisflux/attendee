"""
Tests for the Azure OpenAI credential type (create, encrypted round-trip, defaults, validation).
"""

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from bots.models import DEFAULT_AZURE_OPENAI_API_VERSION, Credentials, Organization, Project

User = get_user_model()

AZURE = Credentials.CredentialTypes.AZURE_OPENAI
CONTAINER_ID = "azure-openai-credentials-container"


class AzureOpenAICredentialsTest(TestCase):
    """Create/save/fetch an Azure OpenAI credential the same way as the other providers."""

    def setUp(self):
        self.client = Client()
        self.organization = Organization.objects.create(name="Test Organization")
        self.user = User.objects.create_user(username="testuser", email="test@example.com", password="testpass123", organization=self.organization)
        self.project = Project.objects.create(name="Test Project", organization=self.organization)

    def _post(self, data):
        payload = {"credential_type": AZURE, "csrfmiddlewaretoken": "test-token"}
        payload.update(data)
        return self.client.post(reverse("projects:create-credentials", args=[self.project.object_id]), payload)

    def _full_payload(self):
        return {
            "endpoint": "https://res.openai.azure.com",
            "deployment": "gpt-4o",
            "api_key": "secret-key-123",
            "api_version": "2025-01-01",
        }

    def test_set_get_round_trip_encrypts_the_four_field_dict(self):
        """set_credentials/get_credentials returns the exact dict and stores it encrypted."""
        cred = Credentials.objects.create(project=self.project, credential_type=AZURE)
        data = self._full_payload()
        cred.set_credentials(data)

        self.assertEqual(cred.get_credentials(), data)
        # Stored blob is encrypted, not the plaintext key.
        self.assertNotIn(b"secret-key-123", bytes(cred._encrypted_data))

    def test_create_success_stores_all_fields(self):
        self.client.force_login(self.user)
        response = self._post(self._full_payload())

        self.assertEqual(response.status_code, 200)
        self.assertIn(CONTAINER_ID, response.content.decode())

        cred = Credentials.objects.get(project=self.project, credential_type=AZURE)
        self.assertEqual(cred.get_credentials(), self._full_payload())

    def test_api_version_defaults_when_omitted(self):
        self.client.force_login(self.user)
        data = self._full_payload()
        del data["api_version"]
        response = self._post(data)

        self.assertEqual(response.status_code, 200)
        cred = Credentials.objects.get(project=self.project, credential_type=AZURE)
        self.assertEqual(cred.get_credentials()["api_version"], DEFAULT_AZURE_OPENAI_API_VERSION)

    def test_missing_required_field_rejected(self):
        """Each of endpoint / deployment / api_key is required → 400 with no usable credential stored.

        Note: the shared handler calls get_or_create before validating, so a failed submit can leave
        an empty stub row (no encrypted data) — inherited behavior across all credential types. What
        matters is that no *usable* credential is persisted (get_credentials() stays None).
        """
        self.client.force_login(self.user)
        for missing in ("endpoint", "deployment", "api_key"):
            data = self._full_payload()
            del data[missing]
            response = self._post(data)
            self.assertEqual(response.status_code, 400, msg=f"missing {missing}")
        cred = Credentials.objects.filter(project=self.project, credential_type=AZURE).first()
        self.assertTrue(cred is None or cred.get_credentials() is None)

    def test_edit_overwrites_and_keeps_single_row(self):
        """Re-saving (get_or_create) updates the value and never creates a duplicate row."""
        self.client.force_login(self.user)
        self._post(self._full_payload())

        updated = self._full_payload()
        updated["deployment"] = "gpt-4o-mini"
        self._post(updated)

        rows = Credentials.objects.filter(project=self.project, credential_type=AZURE)
        self.assertEqual(rows.count(), 1)
        self.assertEqual(rows.first().get_credentials()["deployment"], "gpt-4o-mini")

    def test_unauthorized_user_cannot_create(self):
        other_org = Organization.objects.create(name="Other Organization")
        other_user = User.objects.create_user(username="otheruser", email="other@example.com", password="testpass123", organization=other_org)
        self.client.force_login(other_user)

        response = self._post(self._full_payload())
        self.assertIn(response.status_code, [403, 302, 404])
        self.assertFalse(Credentials.objects.filter(project=self.project, credential_type=AZURE).exists())

    def test_credentials_page_renders_azure_card_and_context(self):
        """The credentials page GET wires the Azure card + its context (masked key, saved values)."""
        self.client.force_login(self.user)
        Credentials.objects.create(project=self.project, credential_type=AZURE).set_credentials(self._full_payload())

        response = self.client.get(reverse("projects:project-credentials", args=[self.project.object_id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["azure_openai_credential_type"], AZURE)
        self.assertEqual(response.context["azure_openai_credentials"], self._full_payload())
        body = response.content.decode()
        self.assertIn(CONTAINER_ID, body)
        self.assertIn("•••••••••123", body)  # api_key masked to last 3

    def test_delete_azure_credential(self):
        """Deleting works with no type-specific code and returns to the 'Add' state."""
        self.client.force_login(self.user)
        Credentials.objects.create(project=self.project, credential_type=AZURE).set_credentials(self._full_payload())

        response = self.client.post(reverse("projects:delete-credentials", args=[self.project.object_id]), {"credential_type": AZURE, "csrfmiddlewaretoken": "test-token"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(CONTAINER_ID, response.content.decode())
        self.assertFalse(Credentials.objects.filter(project=self.project, credential_type=AZURE).exists())
