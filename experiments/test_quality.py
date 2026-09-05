from ai_engine.quality import evaluate_registration_quality


result = evaluate_registration_quality(
    total_matches=19,
    inlier_count=5,
)

print("========== LUCAS QUALITY GATE ==========")
print("Registration:", "ACCEPTED" if result["success"] else "REJECTED")
print("Reason:", result["reason"])
print("Inliers:", result["inlier_count"])
print("Inlier ratio:", f"{result['inlier_ratio']:.3f}")
print("========================================")