"""
Test authentication and authorization
"""
import pytest
from fastapi.testclient import TestClient

class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_login_success(self, client: TestClient, admin_user):
        """Test successful login"""
        response = client.post("/api/auth/login", json={
            "username": "test_admin",
            "password": "test_password"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "test_admin"
        assert data["user"]["role"] == "admin"
    
    def test_login_invalid_credentials(self, client: TestClient, admin_user):
        """Test login with invalid credentials"""
        response = client.post("/api/auth/login", json={
            "username": "test_admin",
            "password": "wrong_password"
        })
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_nonexistent_user(self, client: TestClient):
        """Test login with nonexistent user"""
        response = client.post("/api/auth/login", json={
            "username": "nonexistent",
            "password": "password"
        })
        
        assert response.status_code == 401
    
    def test_get_current_user(self, client: TestClient, auth_headers_admin):
        """Test getting current user info"""
        response = client.get("/api/auth/me", headers=auth_headers_admin)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["username"] == "test_admin"
        assert data["role"] == "admin"
    
    def test_get_current_user_no_token(self, client: TestClient):
        """Test getting current user without token"""
        response = client.get("/api/auth/me")
        
        assert response.status_code == 401
    
    def test_register_user_admin(self, client: TestClient, auth_headers_admin):
        """Test user registration by admin"""
        user_data = {
            "username": "new_user",
            "email": "newuser@test.com",
            "password": "password123",
            "full_name": "New User",
            "role": "doctor"
        }
        
        response = client.post("/api/auth/register", json=user_data, headers=auth_headers_admin)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["username"] == "new_user"
        assert data["role"] == "doctor"
    
    def test_register_user_non_admin(self, client: TestClient, auth_headers_doctor):
        """Test user registration by non-admin (should fail)"""
        user_data = {
            "username": "new_user2",
            "email": "newuser2@test.com",
            "password": "password123",
            "role": "viewer"
        }
        
        response = client.post("/api/auth/register", json=user_data, headers=auth_headers_doctor)
        
        assert response.status_code == 403
    
    def test_list_users_admin(self, client: TestClient, auth_headers_admin):
        """Test listing users as admin"""
        response = client.get("/api/auth/users", headers=auth_headers_admin)
        
        assert response.status_code == 200
        users = response.json()
        
        assert isinstance(users, list)
        assert len(users) > 0
    
    def test_list_users_non_admin(self, client: TestClient, auth_headers_doctor):
        """Test listing users as non-admin (should fail)"""
        response = client.get("/api/auth/users", headers=auth_headers_doctor)
        
        assert response.status_code == 403

class TestAuthorization:
    """Test authorization and permissions"""
    
    def test_permission_checking(self, client: TestClient, auth_headers_doctor):
        """Test permission-based access"""
        # Doctor should be able to view models
        response = client.get("/api/models/", headers=auth_headers_doctor)
        assert response.status_code == 200
        
        # Doctor should not be able to manage models
        response = client.post("/api/models/refresh", headers=auth_headers_doctor)
        assert response.status_code == 403
    
    def test_role_based_access(self, client: TestClient, auth_headers_admin, auth_headers_doctor):
        """Test role-based access control"""
        # Admin can access everything
        response = client.get("/api/auth/users", headers=auth_headers_admin)
        assert response.status_code == 200
        
        # Doctor cannot access user management
        response = client.get("/api/auth/users", headers=auth_headers_doctor)
        assert response.status_code == 403