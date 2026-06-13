import json

from app.domain import models


def _make_notes(count=2):
    return [
        {
            "time": index * 0.5,
            "duration": 0.5,
            "pitch": 64 + index,
            "string": 1,
            "fret": index,
        }
        for index in range(count)
    ]


def _create_saved_tab(test_db, user_id, filename="song.wav", note_count=2, content=None):
    if content is None:
        content = {
            "tablature": _make_notes(note_count),
            "tablature_beginner": _make_notes(note_count),
        }

    tab = models.Tablature(
        filename=filename,
        json_content=json.dumps(content),
        user_id=user_id,
    )
    test_db.add(tab)
    test_db.commit()
    test_db.refresh(tab)
    return tab


def test_history_is_empty_for_new_user(client, auth_headers):
    response = client.get("/history", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == []


def test_history_returns_saved_tablatures_for_current_user(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    _create_saved_tab(test_db, user.id, filename="first.wav")
    _create_saved_tab(test_db, user.id, filename="second.wav")

    response = client.get("/history", headers=auth_headers)

    body = response.json()
    assert response.status_code == 200
    assert len(body) == 2
    assert body[0]["filename"] == "second.wav"
    assert "preview" in body[0]


def test_history_preview_contains_only_first_five_notes(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    _create_saved_tab(test_db, user.id, filename="long.wav", note_count=7)

    response = client.get("/history", headers=auth_headers)

    body = response.json()
    assert response.status_code == 200
    assert len(body[0]["tablature"]) == 7
    assert len(body[0]["preview"]) == 5


def test_history_does_not_return_other_user_tablatures(client, test_db, registered_user, auth_headers):
    current_user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    other_user = models.User(username="other", password_hash="hash")
    test_db.add(other_user)
    test_db.commit()
    test_db.refresh(other_user)

    _create_saved_tab(test_db, current_user.id, filename="mine.wav")
    _create_saved_tab(test_db, other_user.id, filename="foreign.wav")

    response = client.get("/history", headers=auth_headers)

    filenames = [item["filename"] for item in response.json()]
    assert response.status_code == 200
    assert filenames == ["mine.wav"]


def test_history_with_invalid_saved_json_does_not_crash(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    tab = models.Tablature(filename="broken.wav", json_content="not-json", user_id=user.id)
    test_db.add(tab)
    test_db.commit()

    response = client.get("/history", headers=auth_headers)

    body = response.json()
    assert response.status_code == 200
    assert body[0]["filename"] == "broken.wav"
    assert body[0]["tablature"] == []
    assert body[0]["tablature_beginner"] == []
    assert body[0]["preview"] == []


def test_delete_saved_tablature_success(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    tab = _create_saved_tab(test_db, user.id)

    response = client.delete(f"/history/{tab.id}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["id"] == tab.id
    assert response.json()["message"] == "Tablature deleted successfully"
    assert test_db.query(models.Tablature).filter(models.Tablature.id == tab.id).first() is None


def test_deleted_tablature_no_longer_appears_in_history(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    tab = _create_saved_tab(test_db, user.id, filename="delete_me.wav")

    delete_response = client.delete(f"/history/{tab.id}", headers=auth_headers)
    history_response = client.get("/history", headers=auth_headers)

    assert delete_response.status_code == 200
    assert history_response.status_code == 200
    assert history_response.json() == []


def test_delete_missing_tablature_returns_404(client, auth_headers):
    response = client.delete("/history/9999", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Tablature not found"


def test_delete_foreign_tablature_returns_404(client, test_db, auth_headers):
    other_user = models.User(username="other", password_hash="hash")
    test_db.add(other_user)
    test_db.commit()
    test_db.refresh(other_user)
    foreign_tab = _create_saved_tab(test_db, other_user.id)

    response = client.delete(f"/history/{foreign_tab.id}", headers=auth_headers)

    assert response.status_code == 404


def test_export_original_pdf_success(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    tab = _create_saved_tab(test_db, user.id, filename="solo.wav")

    response = client.get(f"/history/{tab.id}/export-pdf?variant=original", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert "solo_original_tab.pdf" in response.headers["content-disposition"]


def test_export_beginner_pdf_success(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    tab = _create_saved_tab(test_db, user.id, filename="solo.wav")

    response = client.get(f"/history/{tab.id}/export-pdf?variant=beginner", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert "solo_beginner_tab.pdf" in response.headers["content-disposition"]


def test_export_invalid_variant_returns_422(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    tab = _create_saved_tab(test_db, user.id)

    response = client.get(f"/history/{tab.id}/export-pdf?variant=wrong", headers=auth_headers)

    assert response.status_code == 422


def test_export_missing_tablature_returns_404(client, auth_headers):
    response = client.get("/history/9999/export-pdf?variant=original", headers=auth_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Tablature not found"


def test_export_foreign_tablature_returns_404(client, test_db, auth_headers):
    other_user = models.User(username="other", password_hash="hash")
    test_db.add(other_user)
    test_db.commit()
    test_db.refresh(other_user)
    foreign_tab = _create_saved_tab(test_db, other_user.id)

    response = client.get(f"/history/{foreign_tab.id}/export-pdf?variant=original", headers=auth_headers)

    assert response.status_code == 404


def test_export_pdf_with_empty_beginner_notes_still_returns_pdf(client, test_db, registered_user, auth_headers):
    user = test_db.query(models.User).filter(models.User.username == registered_user["username"]).first()
    tab = _create_saved_tab(
        test_db,
        user.id,
        filename="empty_beginner.wav",
        content={"tablature": _make_notes(2), "tablature_beginner": []},
    )

    response = client.get(f"/history/{tab.id}/export-pdf?variant=beginner", headers=auth_headers)

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
