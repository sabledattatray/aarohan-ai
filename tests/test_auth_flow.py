"""Automated tests for mandatory login, Email+OTP authentication, Google One Tap, and Theme Engine."""

from starlette.testclient import TestClient

from aarohan.api import create_app


def test_auth_otp_flow_and_google_one_tap():
    app = create_app()
    client = TestClient(app)

    # 1. Invalid email format rejection
    resp = client.post("/api/auth/send-otp", json={"email": "not-an-email"})
    assert resp.status_code == 400

    # 2. Send OTP to valid email
    test_email = "datta.sable@gmail.com"
    send_resp = client.post("/api/auth/send-otp", json={"email": test_email})
    assert send_resp.status_code == 200
    send_data = send_resp.json()
    assert send_data["success"] is True
    assert "dev_otp" in send_data
    otp = send_data["dev_otp"]
    assert len(otp) == 6

    # 3. Verify with wrong OTP
    bad_verify = client.post("/api/auth/verify-otp", json={"email": test_email, "otp": "000000"})
    assert bad_verify.status_code == 400

    # 4. Verify with correct OTP
    good_verify = client.post("/api/auth/verify-otp", json={"email": test_email, "otp": otp})
    assert good_verify.status_code == 200
    auth_data = good_verify.json()
    assert "token" in auth_data
    assert auth_data["user"]["email"] == test_email
    token = auth_data["token"]

    # 5. /api/auth/me with valid bearer token (GET)
    me_get = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_get.status_code == 200
    assert me_get.json()["email"] == test_email

    # 6. Google Sign-In / One Tap endpoint
    google_resp = client.post(
        "/api/auth/google",
        json={"email": "datta@google.com", "name": "Datta Sable", "credential": "dummy.header.payload"},
    )
    assert google_resp.status_code == 200
    g_data = google_resp.json()
    assert "token" in g_data
    assert g_data["user"]["name"] == "Datta Sable"

    # 7. Logout endpoint
    logout_resp = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_resp.status_code == 200
    assert logout_resp.json()["success"] is True


def test_studio_html_mandatory_auth_and_light_theme_conformance():
    app = create_app()
    client = TestClient(app)

    resp = client.get("/app")
    assert resp.status_code == 200
    html = resp.text

    # Mandatory Auth Gate Modal & Elements
    assert 'id="auth-gate-modal"' in html
    assert "Continue with Google" in html
    assert "707439992057-j87plivvk29u7nq35l1j7sqdraoqhv5u.apps.googleusercontent.com" in html
    assert 'id="auth-email-input"' in html
    assert 'id="auth-otp-input"' in html
    assert "sendEmailOTPAction" in html
    assert "verifyEmailOTPAction" in html
    assert "logoutAction" in html
    assert "checkAuthSession" in html

    # Light Theme Complete Overrides in <style>
    assert "html.light body" in html
    assert "html.light aside#sidebar" in html
    assert "html.light main" in html
    assert "html.light textarea#chat-input" in html
    assert "html.light #chat-composer-container" in html
    assert "html.light .bg-\\[\\#212121\\]" in html
    assert "html.light .border-\\[\\#383838\\]" in html
