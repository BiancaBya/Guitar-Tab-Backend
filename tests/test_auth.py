from jose import jwt

from app import auth, models


def test_register_user_success(client):
    response = client.post(
        "/register",
        json={"username": "maria", "password": "secret123"},
    )

    assert response.status_code == 201
    assert response.json()["message"] == "User created successfully"


def test_register_duplicate_username_returns_error(client):
    payload = {"username": "maria", "password": "secret123"}

    first_response = client.post("/register", json=payload)
    second_response = client.post("/register", json=payload)

    assert first_response.status_code == 201
    assert second_response.status_code == 400
    assert second_response.json()["detail"] == "Username already registered"


def test_registered_password_is_not_saved_as_plain_text(client, test_db):
    client.post("/register", json={"username": "maria", "password": "secret123"})

    user = test_db.query(models.User).filter(models.User.username == "maria").first()

    assert user is not None
    assert user.password_hash != "secret123"
    assert auth.verify_password("secret123", user.password_hash)


def test_password_hash_can_be_verified():
    password = "parola-mea"

    hashed_password = auth.get_password_hash(password)

    assert hashed_password != password
    assert auth.verify_password(password, hashed_password)
    assert not auth.verify_password("alta-parola", hashed_password)


def test_login_with_valid_credentials_returns_token(client, registered_user):
    response = client.post(
        "/token",
        data={
            "username": registered_user["username"],
            "password": registered_user["password"],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str)
    assert len(body["access_token"]) > 20


def test_login_with_wrong_password_is_rejected(client, registered_user):
    response = client.post(
        "/token",
        data={
            "username": registered_user["username"],
            "password": "gresit",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


def test_login_with_unknown_username_is_rejected(client):
    response = client.post(
        "/token",
        data={"username": "nu_exista", "password": "parola123"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


def test_created_token_contains_username_in_subject():
    token = auth.create_access_token(data={"sub": "ana"})

    payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])

    assert payload["sub"] == "ana"
    assert "exp" in payload


def test_me_with_valid_token_returns_current_user(client, auth_headers):
    response = client.get("/me", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["username"] == "ana"
    assert "id" in response.json()


def test_me_without_token_is_rejected(client):
    response = client.get("/me")

    assert response.status_code == 401


def test_me_with_invalid_token_is_rejected(client):
    response = client.get("/me", headers={"Authorization": "Bearer token-invalid"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"


def test_history_without_token_is_rejected(client):
    response = client.get("/history")

    assert response.status_code == 401


def test_export_without_token_is_rejected(client):
    response = client.get("/history/1/export-pdf?variant=original")

    assert response.status_code == 401


def test_delete_without_token_is_rejected(client):
    response = client.delete("/history/1")

    assert response.status_code == 401


def test_predict_without_token_is_rejected(client, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio content")

    with audio_path.open("rb") as f:
        response = client.post(
            "/predict-tab/",
            files={"file": ("sample.wav", f, "audio/wav")},
        )

    assert response.status_code == 401
