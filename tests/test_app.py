from http import HTTPStatus

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert data['status'] == 'running'
    assert data['message'] == 'Welcome to Deep Learning SHOWCASE!'


def test_root():
    response = client.get('/welcome')
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {'message': 'Welcome to Deep Learning SHOWCASE!'}
