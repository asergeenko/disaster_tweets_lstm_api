from fastapi.testclient import TestClient
from deploy.api import app

client = TestClient(app)


def test_rest_service_e2e():
    response = client.get("/predict/Damage%20to%20school%20bus%20on%2080%20in%20multi%20car%20crash")
    assert response.status_code == 200
    assert response.json() == {"message":"Damage to school bus on 80 in multi car crash","disaster_tweet":1}
