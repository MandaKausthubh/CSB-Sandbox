1. POST /api/auth/login - User login: Expects data in the form: fastapi.security.OAuth2PasswordRequestForm
2. LogOut expects things in the form UserDB
3. GET /api/auth/me - Get current user: Expected request format: 
```
class UserDB(Base):
    __tablename__ = "users"
    userId = Column(Integer, primary_key=True, index=True)
    userName = Column(String, nullable=False)
    userEmail = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

```
4. POST /api/analysis/create - Create new analysis, expected request format:
```
class AnalysisCreate(BaseModel):
    content: str
    contentType: str = "text"
    targetAudience: List[str]
    platform: List[str]
    region: List[str]
    sponsors: List[str] = []
    project_id: Optional[int] = None  # still supported

```
5. GET /api/analysis/history - Get analysis history
```
@router.get("/history")
def get_analysis_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
)
```
where the Query object is from fastapi   
6. GET /api/analysis/:id - Get specific analysis, expected request body:
```
@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
```
7. DELETE /api/analysis/:id - Delete analysis
```
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
```
