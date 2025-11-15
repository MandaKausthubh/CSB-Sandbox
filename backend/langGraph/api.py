# pyright: reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportAssignmentType=false
# pyright: ignoreall
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import json
from flask import jsonify

from langGraph.run_analysis import run_graph_analysis
from database.database import get_db
from core.auth import get_current_user
from models.analysis import AnalysisCreate, AnalysisOut
from models.analysis import Analysis
from langGraph.analysis import run_analysis

router = APIRouter(prefix="/analysis", tags=["Analysis"])

@router.post("/create")
def create_analysis(
    payload: AnalysisCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not payload.content:
        return {
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": "Content is required"
        }

    params = {
        "contentType": payload.contentType,
        "targetAudience": payload.targetAudience,
        "platform": payload.platform,
        "region": payload.region,
        "sponsors": payload.sponsors
    }

    # Store in DB
    analysis = Analysis(
        owner_id=user.userId,
        project_id=payload.project_id if payload.project_id else None,
        title=None,
        input_text=payload.content,
        parameters=json.dumps(params),
        status="processing",
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Run AI pipeline
    result = run_analysis(payload.content, params)

    analysis.result = json.dumps(result)
    analysis.score = result["score"]
    analysis.status = result["complianceStatus"]
    db.commit()
    db.refresh(analysis)

    # Build final FE-compatible response
    return ({
        "success": True,
        "analysis": {
            "id": analysis.id,
            "userId": analysis.owner_id,
            "content": payload.content,
            "contentType": payload.contentType,
            "targetAudience": payload.targetAudience,
            "platform": payload.platform,
            "region": payload.region,
            "sponsors": payload.sponsors,
            "score": result["score"],
            "complianceStatus": result["complianceStatus"],
            "violations": result["violations"],
            "suggestions": result["suggestions"],
            "aiEnhancedScript": result["aiEnhancedScript"],
            "createdAt": analysis.created_at,
        }
    })


@router.get("/history")
def get_analysis_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # Total count for pagination
    total = db.query(Analysis).filter(
        Analysis.owner_id == user.userId
    ).count()

    # Fetch page
    offset = (page - 1) * limit

    rows = (
        db.query(Analysis)
        .filter(Analysis.owner_id == user.userId)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    analyses_list = []
    for a in rows:
        params = json.loads(a.parameters) if a.parameters else {} # type: ignore
        analysis_dict = {
            "id": a.id,
            "userId": a.owner_id,
            "content": a.input_text,
            "contentType": params.get("contentType", "text"),
            "targetAudience": params.get("targetAudience", []),
            "platform": params.get("platform", []),
            "region": params.get("region", []),
            "score": a.score,
            "complianceStatus": a.status,
            "createdAt": a.created_at,
        }
        analyses_list.append(analysis_dict)

    total_pages = (total + limit - 1) // limit  # ceiling division

    return {
        "success": True,
        "analyses": analyses_list,
        "total": total,
        "page": page,
        "totalPages": total_pages,
    }

@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # Fetch analysis
    a = (
        db.query(Analysis)
        .filter(Analysis.id == analysis_id, Analysis.owner_id == user.userId)
        .first()
    )

    if not a:
        return {
            "success": False,
            "error": "NOT_FOUND",
            "message": "Analysis not found"
        }

    # Parse parameters & results JSON
    params = json.loads(a.parameters) if a.parameters else {} # type: ignore
    result = json.loads(a.result) if a.result else {} # type: ignore

    # Build FE-compatible response object
    analysis_payload = {
        "id": a.id,
        "userId": a.owner_id,
        "content": a.input_text,
        "contentType": params.get("contentType", "text"),
        "targetAudience": params.get("targetAudience", []),
        "platform": params.get("platform", []),
        "region": params.get("region", []),
        "sponsors": params.get("sponsors", []),
        "score": result.get("score", a.score),
        "complianceStatus": result.get("complianceStatus", a.status),
        "violations": result.get("violations", []),
        "suggestions": result.get("suggestions", []),
        "aiEnhancedScript": result.get("aiEnhancedScript", a.input_text),
        "createdAt": a.created_at,
    }

    return {
        "success": True,
        "analysis": analysis_payload
    }


# pyright: ignoreall
@router.delete("/{analysis_id}")
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # Find the analysis, ensure it belongs to the user
    analysis = (
        db.query(Analysis)
        .filter(Analysis.id == analysis_id, Analysis.owner_id == user.userId)
        .first()
    )

    if not analysis:
        return {
            "success": False,
            "error": "NOT_FOUND",
            "message": "Analysis not found"
        }

    db.delete(analysis)
    db.commit()

    return {"success": True}







