# mappings.py
import numpy as np

loan_status_map = {
    "Charged Off": 0,
    "Current": 1,
    "Fully Paid": 2,
    "In Grace Period": 3,
    "Late (31-120 days)": 4
}

grade_map = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5
}

sub_grade_map = {
    "A1": 0, "A2": 1, "A3": 2, "A4": 3, "A5": 4,
    "B1": 5, "B2": 6, "B3": 7, "B4": 8, "B5": 9,
    "C1": 10, "C2": 11, "C3": 12, "C4": 13, "C5": 14,
    "D1": 15, "D2": 16, "D3": 17, "D4": 18, "D5": 19,
    "E1": 20, "E2": 21, "E3": 22, "E4": 23, "E5": 24,
    "F1": 25, "F2": 26, "F3": 27, "F4": 28, "F5": 29
}

home_ownership_map = {
    "RENT": 0,
    "OWN": 1,
    "MORTGAGE": 2
}

verification_status_map = {
    "Not Verified": 0,
    "Source Verified": 1,
    "Verified": 2
}

pymnt_plan_map = {
    "n": 0,
    "y": 1
}

debt_settlement_flag_map = {
    "N": 0,
    "Y": 1
}

settlement_status_map = {
    "ACTIVE": 0,
    "BROKEN": 1,
    "COMPLETE": 2
}

initial_list_status_map = {
    "w": 0,
    "f": 1
}

application_type_map = {
    "Individual": 0,
    "Joint App": 1
}

term_map = {
    "36 months": 36,
    "60 months": 60
}

verification_status_joint_map = {
    "Not Verified": 0,
    "Source Verified": 1,
    "Verified": 2,
    np.nan: -1  # 'Not Applicable' case
}

purpose_map = {
    "debt_consolidation": 0,
    "small_business": 1,
    "home_improvement": 2,
    "major_purchase": 3,
    "credit_card": 4,
    "other": 5,
    "house": 6,
    "vacation": 7,
    "car": 8,
    "medical": 9,
    "moving": 10
}
