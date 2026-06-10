# test_meeting_utils.py
import base64
import json
import unittest

from bots.meeting_url_utils import MeetingTypes, canonical_meeting_id, domain_and_subdomain_from_url, meeting_type_from_url, normalize_meeting_url, parse_zoom_join_url, parse_zoom_registrant_token, root_domain_from_url


class TestMeetingUrlUtils(unittest.TestCase):
    def test_root_domain_from_url(self):
        self.assertEqual(root_domain_from_url("https://meet.google.com/abc-defg-hij"), "google.com")
        self.assertEqual(root_domain_from_url("https://teams.microsoft.com/l/meetup-join/..."), "microsoft.com")
        self.assertEqual(root_domain_from_url("https://zoom.us/j/123456789"), "zoom.us")
        self.assertIsNone(root_domain_from_url(""))
        self.assertIsNone(root_domain_from_url(None))

    def test_domain_and_subdomain_from_url(self):
        self.assertEqual(domain_and_subdomain_from_url("https://meet.google.com/abc-defg-hij"), "meet.google.com")
        self.assertEqual(domain_and_subdomain_from_url("https://teams.microsoft.com/l/meetup-join/..."), "teams.microsoft.com")
        self.assertEqual(domain_and_subdomain_from_url("https://zoom.us/j/123456789"), ".zoom.us")
        self.assertIsNone(domain_and_subdomain_from_url(""))
        self.assertIsNone(domain_and_subdomain_from_url(None))

    def test_teams_live_urls(self):
        self.assertEqual(meeting_type_from_url("https://teams.live.com/meet/9876543210?p=abc"), MeetingTypes.TEAMS)
        self.assertEqual(meeting_type_from_url("https://teams.microsoft.com/meet/9876543210?p=qHDqtvFSfIg1rT"), MeetingTypes.TEAMS)
        self.assertEqual(meeting_type_from_url("https://teams.live.com/meet/9876543210?p=abc>"), MeetingTypes.TEAMS)
        self.assertEqual(normalize_meeting_url("https://teams.live.com/meet/9876543210?p=abc>")[1], "https://teams.live.com/meet/9876543210?p=abc")

    def test_teams_meetup_join_urls(self):
        teams_url_with_trailing_carat = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_NjnnnnnnnnnnnnnnnnnnnnnnnnnnMzctODQwYTBmMDQ4MzQ2%40thread.v2/0?context=%7b%22Tid%22%3a%22b8291b4b-aaaa-bbbb-8a00-9d5fc37b9a77%22%2c%22Oid%22%3a%22216d2e11-45cc-9999-9689-d05554e5c1d1%22%7d>"
        self.assertEqual(meeting_type_from_url(teams_url_with_trailing_carat), MeetingTypes.TEAMS)
        self.assertEqual(normalize_meeting_url(teams_url_with_trailing_carat)[1], 'https://teams.microsoft.com/l/meetup-join/19:meeting_NjnnnnnnnnnnnnnnnnnnnnnnnnnnMzctODQwYTBmMDQ4MzQ2@thread.v2/0?context={"Tid":"b8291b4b-aaaa-bbbb-8a00-9d5fc37b9a77","Oid":"216d2e11-45cc-9999-9689-d05554e5c1d1"}')

    def test_normalize_meeting_url(self):
        self.assertEqual(normalize_meeting_url("https://zoom.us/j/123456789")[1], "https://zoom.us/j/123456789")
        self.assertEqual(normalize_meeting_url("zoom.us/j/123456789")[1], "https://zoom.us/j/123456789")
        self.assertEqual(normalize_meeting_url("https://meet.google.com/abc-defg-hij")[1], "https://meet.google.com/abc-defg-hij")

        self.assertEqual(normalize_meeting_url("meet.google.com/abc-defg-hij")[1], "https://meet.google.com/abc-defg-hij")
        self.assertEqual(normalize_meeting_url("https://teams.microsoft.com/l/meetup-join/19%3ameeting_NjnnnnnnnnnnnnnnnnnnnnnnnnnnMzctODQwYTBmMDQ4MzQ2%40thread.v2/0?context=%7b%22Tid%22%3a%22b8291b4b-aaaa-bbbb-8a00-9d5fc37b9a77%22%2c%22Oid%22%3a%22216d2e11-45cc-9999-9689-d05554e5c1d1%22%7d>")[1], 'https://teams.microsoft.com/l/meetup-join/19:meeting_NjnnnnnnnnnnnnnnnnnnnnnnnnnnMzctODQwYTBmMDQ4MzQ2@thread.v2/0?context={"Tid":"b8291b4b-aaaa-bbbb-8a00-9d5fc37b9a77","Oid":"216d2e11-45cc-9999-9689-d05554e5c1d1"}')
        self.assertEqual(normalize_meeting_url("https://teams.microsoft.com/meet/999999999999?p=aaaaalRE2XBPsAr00W")[1], "https://teams.microsoft.com/meet/999999999999?p=aaaaalRE2XBPsAr00W")

        self.assertEqual(normalize_meeting_url("https://teams.live.com/meet/999999999999?p=aaaaalRE2XBPsAr00W")[1], "https://teams.live.com/meet/999999999999?p=aaaaalRE2XBPsAr00W")
        self.assertEqual(normalize_meeting_url("https://teams.microsoft.com/dl/launcher/launcher.html?url=/_#/l/meetup-join/19:meeting_NDQ3Y2Q1NDEtY2I5Ni00MzEyLTgzfffffffffffffffffffffffffff@thread.v2/0?context=%7b%22Tid%22%3a%22b00367e2-aaaa-bbbb-cccc-7245d45c0947%22%2c%22Oid%22%3a%2266666666-3a9d-441a-892d-55555555555%22%7d&anon=true&type=meetup-join&deeplinkId=5044eb28-9d33-4309-bf23-3aa618a9e6b0&directDl=true&msLaunch=true&enableMobilePage=true&suppressPrompt=true")[1], 'https://teams.microsoft.com/l/meetup-join/19:meeting_NDQ3Y2Q1NDEtY2I5Ni00MzEyLTgzfffffffffffffffffffffffffff@thread.v2/0?context={"Tid":"b00367e2-aaaa-bbbb-cccc-7245d45c0947","Oid":"66666666-3a9d-441a-892d-55555555555"}')
        coord_json = {"conversationId": "19:meeting_ffffffffffffffffffffffffffffffffffffffffffffffff@thread.v2", "tenantId": "ffffffff-ffff-fff-ffff-6866ff300052", "organizerId": "ffffffff-ffff-4085-acbd-ffffffffffff", "messageId": "0"}
        fake_coord = base64.b64encode(json.dumps(coord_json).encode("utf-8"))
        # Drop trailing equals
        fake_coord = fake_coord.decode("utf-8").rstrip("=")
        self.assertEqual(normalize_meeting_url(f"https://teams.microsoft.com/light-meetings/launch?agent=web&version=25072001100&coords={fake_coord}%3D&deeplinkId=0ac0a241-11111-111-8ddc-4a4b333dd286&correlationId=1111111-1111-1111-1111-11111111111")[1], "https://teams.microsoft.com/l/meetup-join/" + coord_json["conversationId"] + f"/0?context={json.dumps({'Tid': coord_json['tenantId'], 'Oid': coord_json['organizerId']}, separators=(',', ':'))}")

        self.assertEqual(normalize_meeting_url("https://zoom.us/w/123456789?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.DQkAAAAATESTTOKEN1234567890AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")[1], "https://zoom.us/w/123456789?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.DQkAAAAATESTTOKEN1234567890AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    def test_meeting_type_from_url(self):
        self.assertEqual(meeting_type_from_url("https://zoom.us/j/123456789"), MeetingTypes.ZOOM)
        self.assertEqual(meeting_type_from_url("https://zoom.us/j/test"), None)
        self.assertEqual(meeting_type_from_url("https://meet.google.com/abc-defg-hij"), MeetingTypes.GOOGLE_MEET)
        self.assertEqual(meeting_type_from_url("https://teams.microsoft.com/l/meetup-join/19%3ameeting_ABCDEFGHIJKLMNOPQRSTUVWXYZ@thread.v2/0"), None)
        self.assertEqual(meeting_type_from_url("https://teams.live.com/l/meetup-join/19%3ameeting_1234567890ABCDEFGHIJK@thread.v2/0?context=xyz"), None)
        self.assertEqual(meeting_type_from_url("https://teams.live.com/meet/1234567890?p=fffffffffff"), MeetingTypes.TEAMS)
        self.assertEqual(meeting_type_from_url("https://teams.live.com/meet/9876543210?p=abc"), MeetingTypes.TEAMS)
        self.assertIsNone(meeting_type_from_url("https://example.com"))
        self.assertIsNone(meeting_type_from_url(""))
        self.assertIsNone(meeting_type_from_url(None))
        self.assertEqual(meeting_type_from_url("https://teams.microsoft.com/l/meetup-join/19%3ameeting_OTA0nTDmYgItYTlTti00MmRkLTgxODItZGFmNWVmNTJmOGQ4%40thread.v2"), None)

    def test_zoom_com_urls(self):
        # zoom.com should be recognized as a Zoom meeting
        self.assertEqual(meeting_type_from_url("https://zoom.com/j/123456789"), MeetingTypes.ZOOM)
        self.assertEqual(meeting_type_from_url("https://us02web.zoom.com/j/123456789"), MeetingTypes.ZOOM)
        self.assertEqual(meeting_type_from_url("https://zoom.com/w/987654321"), MeetingTypes.ZOOM)

        # zoom.com should still require an integer meeting ID
        self.assertIsNone(meeting_type_from_url("https://zoom.com/j/test"))

        # The bare zoom.com netloc should be normalized to zoom.us
        self.assertEqual(normalize_meeting_url("https://zoom.com/j/123456789")[1], "https://zoom.us/j/123456789")
        self.assertEqual(normalize_meeting_url("zoom.com/j/123456789")[1], "https://zoom.us/j/123456789")

        # Subdomains of zoom.com should have their suffix swapped to zoom.us
        self.assertEqual(normalize_meeting_url("https://us02web.zoom.com/j/123456789")[1], "https://us02web.zoom.us/j/123456789")
        self.assertEqual(normalize_meeting_url("https://us05web.zoom.com/w/987654321")[1], "https://us05web.zoom.us/w/987654321")

        # The pwd and tk query parameters should be preserved when normalizing zoom.com to zoom.us
        self.assertEqual(
            normalize_meeting_url("https://zoom.com/w/123456789?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.DQkAAAAATESTTOKEN1234567890AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")[1],
            "https://zoom.us/w/123456789?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.DQkAAAAATESTTOKEN1234567890AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        )

        # parse_zoom_join_url and parse_zoom_registrant_token should still work on zoom.com URLs
        meeting_id, password = parse_zoom_join_url("https://zoom.com/w/111222333?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.TESTTOKEN")
        registrant_token = parse_zoom_registrant_token("https://zoom.com/w/111222333?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.TESTTOKEN")
        self.assertEqual(meeting_id, "111222333")
        self.assertEqual(password, "AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1")
        self.assertEqual(registrant_token, "ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.TESTTOKEN")

    def test_parse_zoom_webinar_url(self):
        meeting_id, password = parse_zoom_join_url("https://zoom.us/w/111222333?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.DQkAAAAATESTTOKEN1234567890AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        registrant_token = parse_zoom_registrant_token("https://zoom.us/w/111222333?pwd=AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.DQkAAAAATESTTOKEN1234567890AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        self.assertEqual(meeting_id, "111222333")
        self.assertEqual(password, "AbC9xYpQ2LmN7RkT5sVuH4ZbJe1DfG.1")
        self.assertEqual(registrant_token, "ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.DQkAAAAATESTTOKEN1234567890AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")


class TestCanonicalMeetingId(unittest.TestCase):
    """Corpus tests for canonical_meeting_id: every link variant of the same meeting must map to
    the same fingerprint (used to enforce one active bot per meeting per project)."""

    def assert_all_map_to(self, expected_fingerprint, url_variants):
        for url in url_variants:
            self.assertEqual(canonical_meeting_id(url), expected_fingerprint, f"variant did not match: {url}")

    def test_zoom_variants_same_meeting(self):
        self.assert_all_map_to(
            "zoom:84278559171",
            [
                "https://us05web.zoom.us/j/84278559171?pwd=6MKZjG0RFpo6ydbDGW30fRpP7yCTcb.1",
                "https://zoom.us/j/84278559171",
                "https://us05web.zoom.com/j/84278559171?pwd=6MKZjG0RFpo6ydbDGW30fRpP7yCTcb.1",
                # per-registrant tk token must not change the meeting identity
                "https://us05web.zoom.us/j/84278559171?pwd=6MKZjG0RFpo6ydbDGW30fRpP7yCTcb.1&tk=ZkPqN8f2LrS4XyT6wVaE9mHuCdJg5QbA1sDoRtUvWxY.TESTTOKEN",
                "zoom.us/j/84278559171",
                # webinar path variant
                "https://zoom.us/w/84278559171?pwd=6MKZjG0RFpo6ydbDGW30fRpP7yCTcb.1",
            ],
        )

    def test_google_meet_variants_same_meeting(self):
        self.assert_all_map_to(
            "meet:rhg-phjh-vrq",
            [
                "https://meet.google.com/rhg-phjh-vrq",
                "meet.google.com/rhg-phjh-vrq?authuser=0&hs=122",
                "https://meet.google.com/rhg-phjh-vrq?pli=1",
            ],
        )

    def test_teams_meetup_join_variants_same_meeting(self):
        direct = "https://teams.microsoft.com/l/meetup-join/19%3ameeting_NzI4MDc4%40thread.v2/0?context=%7b%22Tid%22%3a%22tenant-111%22%2c%22Oid%22%3a%22organizer-222%22%7d"
        launcher = 'https://teams.microsoft.com/dl/launcher/launcher.html?url=/_#/l/meetup-join/19:meeting_NzI4MDc4@thread.v2/0?context={"Tid":"tenant-111","Oid":"organizer-222"}&type=meetup-join&directDl=true'
        v2 = 'https://teams.microsoft.com/v2/?meetingjoin=true#/l/meetup-join/19:meeting_NzI4MDc4@thread.v2/0?context={"Tid":"tenant-111","Oid":"organizer-222"}&anon=true'
        self.assert_all_map_to("teams:19:meeting_NzI4MDc4@thread.v2:tenant-111:organizer-222", [direct, launcher, v2])

    def test_teams_meet_family_variants_same_meeting(self):
        # passcode (p=) must not change the meeting identity
        self.assert_all_map_to(
            "teams-meet:9512345678901",
            [
                "https://teams.live.com/meet/9512345678901?p=AbCdEfGh123",
                "https://teams.live.com/dl/launcher/launcher.html?url=/_#/meet/9512345678901?p=AbCdEfGh123&anon=true&type=meet",
            ],
        )

    def test_different_meetings_get_different_fingerprints(self):
        self.assertNotEqual(canonical_meeting_id("https://zoom.us/j/84278559171"), canonical_meeting_id("https://zoom.us/j/99999999999"))
        self.assertNotEqual(canonical_meeting_id("https://meet.google.com/rhg-phjh-vrq"), canonical_meeting_id("https://meet.google.com/abc-defg-hij"))
        self.assertNotEqual(canonical_meeting_id("https://teams.live.com/meet/9512345678901?p=a"), canonical_meeting_id("https://teams.live.com/meet/1112223334445?p=a"))

    def test_platforms_never_collide(self):
        fingerprints = [
            canonical_meeting_id("https://zoom.us/j/84278559171"),
            canonical_meeting_id("https://meet.google.com/rhg-phjh-vrq"),
            canonical_meeting_id("https://teams.live.com/meet/9512345678901?p=a"),
        ]
        self.assertEqual(len(set(fingerprints)), len(fingerprints))

    def test_unsupported_or_garbage_urls_return_none(self):
        self.assertIsNone(canonical_meeting_id("https://company.webex.com/meet/room123"))
        self.assertIsNone(canonical_meeting_id("not a url at all"))
        self.assertIsNone(canonical_meeting_id(""))
        self.assertIsNone(canonical_meeting_id(None))

    def test_google_meet_reserved_paths_get_no_fingerprint(self):
        # /lookup/<id> links are DIFFERENT meetings that normalize to the same bare "lookup" path;
        # fingerprinting them would merge unrelated meetings into one dedup slot (false positive).
        self.assertIsNone(canonical_meeting_id("https://meet.google.com/lookup/abc123xyz"))
        self.assertIsNone(canonical_meeting_id("https://meet.google.com/lookup/totally-different"))
        self.assertIsNone(canonical_meeting_id("https://meet.google.com/new"))

    def test_zoom_links_attendee_cannot_normalize_get_no_fingerprint(self):
        # Vanity/personal links have no numeric id; normalize_meeting_url rejects them, so bot
        # creation fails before dedup is ever consulted — fingerprint must agree and return None.
        self.assertIsNone(canonical_meeting_id("https://zoom.us/my/rahulkaushik"))


if __name__ == "__main__":
    unittest.main()
