from fastapi import APIRouter

router = APIRouter(prefix="/options", tags=["Options"])

AUDIENCES = ["Gen Z", "Millennials", "Gen X", "Baby Boomers", "LGBTQ+", "Parents", "Students", "Professionals"]
PLATFORMS = ["YouTube", "Instagram", "TikTok", "Facebook", "Twitter/X", "LinkedIn", "Snapchat", "Pinterest"]
REGIONS = ["IN", "US", "EU", "UK", "CA", "AU", "JP", "BR"]

@router.get("/audiences")
def get_audiences():
    return {"success":True, "audiences": AUDIENCES}

@router.get("/platforms")
def get_platforms():
    return {"success" : True, "platforms": PLATFORMS}

@router.get("/regions")
def get_regions():
    return {"success": True, "regions": REGIONS}
