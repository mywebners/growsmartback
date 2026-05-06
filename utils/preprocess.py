def convert_marks_to_category(mark):
    if mark >= 80:
        return 2
    elif mark >= 60:
        return 1
    else:
        return 0


def map_subjects_to_pslots(marks_dict):
    sorted_subjects = sorted(marks_dict.items(), key=lambda x: x[1], reverse=True)

    p_slots = {}

    for i in range(len(sorted_subjects)):
        subject, marks = sorted_subjects[i]
        p_slots[f"P{i+1}"] = convert_marks_to_category(marks)

    for i in range(len(sorted_subjects), 8):
        p_slots[f"P{i+1}"] = 0

    return p_slots