import pandas as pd
import time
import matplotlib.pyplot as plt
import logging

class Class:
    def __init__(self, name, capacity, special_requirements, subjects, room=None):
        self.name = name
        self.capacity = capacity
        self.special_requirements = special_requirements
        self.timeslots = []
        self.room = room
        self.subjects = subjects

class Teacher:
    def __init__(self, name, available_timeslots):
        self.name = name
        self.available_timeslots = available_timeslots

class Room:
    def __init__(self, name, capacity, available_timeslots):
        self.name = name
        self.capacity = capacity
        self.available_timeslots = available_timeslots
        self.timeslots = []

class Timeslot:
    def __init__(self, day, time):
        self.day = day
        self.time = time

    def __repr__(self):
        return f"{self.day}_{self.time}"

    def __eq__(self, other):
        return isinstance(other, Timeslot) and self.day == other.day and self.time == other.time

    def __hash__(self):
        return hash((self.day, self.time))

class Timetable:
    def __init__(self):
        self.schedule = {}

def read_dataset(file_path="classdataset.txt"):
    classes = []
    teachers = []
    rooms = []
    timeslots = set()
    assignment = {}

    try:
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            # Extracting values from DataFrame rows
            class_name, teacher_name, room_name, day, time = row["Class"], row["Teacher"], row["Room"], row["Day"], row["Time"]

            # Process the data and create instances
            class_obj = Class(class_name, 0, False, [], room=None)
            teacher_obj = Teacher(teacher_name, [])
            # Check if the room already exists
            existing_room = next((room for room in rooms if room.name == room_name), None)
            if existing_room:
                room_obj = existing_room
            else:
                room_obj = Room(room_name, 0, [])
                rooms.append(room_obj)

            timeslot_obj = Timeslot(day, time)

            # Append instances to the respective lists
            classes.append(class_obj)
            teachers.append(teacher_obj)
            timeslots.add(timeslot_obj)

            # Add timeslot to class
            class_obj.timeslots.append(timeslot_obj)

            # Assign teacher, room, and timeslot to the class
            assignment[class_obj] = teacher_obj
            assignment[(class_obj, teacher_obj)] = room_obj
            assignment[class_obj.timeslots[0]] = room_obj

            # Assign timeslot to teacher and room
            assignment[(teacher_obj, class_obj.timeslots[0])] = room_obj

    except Exception as e:
        logging.error(f"Error reading dataset: {e}")
        # Print the line causing the error
        with open(file_path, "r") as file:
            lines = file.readlines()
            logging.error(f"Error in line: {lines[e.__traceback__.tb_lineno - 1]}")

    return classes, teachers, rooms, list(timeslots), assignment
def visualize_schedule(classes, assignment, teachers, rooms):
    print("\nFinal Schedule Visualization\n")
    header = "| {:<15} | {:<15} | {:<15} | {:<15} | {:<15} |"
    divider = "-" * 78
    print(header.format("Class", "Room", "Teacher", "Day", "Time"))
    print(divider)

    for class_obj in classes:
        # Fetch teacher and room assignments
        assigned_teacher = assignment.get(class_obj, None)
        assigned_room = None
        if assigned_teacher:
            assigned_room = assignment.get((class_obj, assigned_teacher), None)

        # Fetch teacher and room names, and timeslot
        teacher_name = assigned_teacher.name if assigned_teacher else "Not Assigned"
        room_name = assigned_room.name if assigned_room else "Not Assigned"
        timeslot = class_obj.timeslots[0] if class_obj.timeslots else "Not Assigned"

        day = timeslot.day if timeslot != "Not Assigned" else "N/A"
        time = timeslot.time if timeslot != "Not Assigned" else "N/A"
        print(header.format(class_obj.name, room_name, teacher_name, day, time))

    print("\nSchedule Visualization Complete\n")

def backtracking_search(csp, use_heuristics=True):
    assignment = {}
    conflicts = []
    result = backtrack(assignment, csp, use_heuristics, conflicts)
    print_conflicts(conflicts)
    return result

def backtrack(assignment, csp, use_heuristics=True, conflicts=[]):
    if len(assignment) == len(csp):
        return assignment

    var = select_unassigned_variable(csp, assignment, use_heuristics)
    logging.debug(f"Trying variable: {var.name}")

    for value in order_domain_values(var, assignment, csp, use_heuristics):
        if is_consistent(var, value, assignment, csp, conflicts):
            assignment[var] = value
            logging.debug(f"Assigned {var.name} to {value.name}")

            if isinstance(var, Class):
                assigned_teacher = assignment.get(var)
                assigned_room = assignment.get((var, assigned_teacher))

                if assigned_room:
                    assigned_room.timeslots.extend(var.timeslots)
                    var.room = assigned_room  # Assign room to class

            result = backtrack(assignment.copy(), csp, use_heuristics, conflicts)
            if result is not None:
                return result

            logging.debug(f"Backtracking on {var.name}")

            if isinstance(var, Class):
                assigned_teacher = assignment.get(var)
                assigned_room = assignment.get((var, assigned_teacher))

                if assigned_room and assigned_room.timeslots:
                    for timeslot in var.timeslots:
                        assigned_room.timeslots.remove(timeslot)
                        del assignment[timeslot]

            del assignment[var]  # Unassign the variable
            del conflicts[-1]

    logging.debug(f"No valid assignment for {var.name}. Conflicts: {conflicts}")
    return None

def select_unassigned_variable(csp, assignment, use_heuristics=True):
    unassigned_vars = [var for var in csp if var not in assignment]
    return mrv_heuristic(unassigned_vars, assignment, csp) if use_heuristics else unassigned_vars[0]

def order_domain_values(var, assignment, csp, use_heuristics=True):
    return lcv_heuristic(csp, var, assignment) if use_heuristics else csp[var]

def is_constraint(var1, value1, var2, value2):
    if isinstance(var1, Class) and isinstance(var2, Teacher):
        return var1.capacity <= value2.capacity
    if isinstance(var1, Class) and isinstance(var2, Room):
        return var1.capacity <= value2.capacity
    if isinstance(var1, Timeslot) and isinstance(var2, Room):
        return True
    if isinstance(var1, Teacher) and isinstance(var2, Timeslot):
        return True
    if isinstance(var1, Room) and isinstance(var2, Timeslot):
        return True
    return False

def is_consistent(var, value, assignment, csp, conflicts):
    logging.debug(f"Checking consistency for {var.name if hasattr(var, 'name') else var} = {value.name}")

    # Check conflicts with neighboring variables
    for neighbor in csp[var]:
        neighbor_value = assignment.get(neighbor)
        if neighbor_value is not None and not constraint(var, value, neighbor, neighbor_value):
            logging.debug(f"Inconsistent with {neighbor.name}")
            conflicts.append((var, neighbor))
            return False

    # Check for conflicts with other assignments
    for assigned_var, assigned_value in assignment.items():
        if assigned_var == var:
            continue
        if not constraint(var, value, assigned_var, assigned_value):
            logging.debug(f"Inconsistent with assigned {assigned_var.name}")
            conflicts.append((var, assigned_var))
            return False

    logging.debug(f"Consistent: {var.name if hasattr(var, 'name') else var} = {value.name}")
    return True

def constraint(var, value, neighbor_value, csp,assignment):
    if var in csp and neighbor_value in assignment:
        return csp[var](value, neighbor_value)
    return True
    

def teacher_constraint(class_obj, teacher_obj, assignment):
    overlap = any(
        assignment[t] == teacher_obj
        for t in class_obj.timeslots
        if t in assignment and assignment[t] is not None
    )
    return not overlap

def room_constraint(class_obj, room_obj, assignment):
    overlap = any(assignment.get(t) == room_obj for t in class_obj.timeslots)
    return not overlap

def capacity_constraint(class_obj, room_obj, assignment):
    assigned_rooms = [
        assigned_value

        for (assigned_var, assigned_value) in assignment.items()
        if isinstance(assigned_var, Class) and isinstance(assigned_value, Room) and assigned_value is not None
    ]
    same_room_count = assigned_rooms.count(room_obj)
    return same_room_count < room_obj.capacity

def preference_constraint(teacher_obj, timeslot_obj):
    return timeslot_obj.day + "_" + timeslot_obj.time in teacher_obj.available_timeslots

def print_conflicts(conflicts):
    if conflicts:
        logging.info("\nConflicts Detected:")
        for conflict in conflicts:
            var1, var2 = conflict
            var1_details = f"{var1.name} (Type: {type(var1).__name__})"
            var2_details = f"{var2.name} (Type: {type(var2).__name__})"
            logging.info(f"Conflict between {var1_details} and {var2_details}")
    else:
        logging.info("\nNo conflicts detected in the algorithm.")


def mrv_heuristic(unassigned_vars, assignment, full_csp):
    return min(
        unassigned_vars,
        key=lambda x: len(getattr(full_csp[x], 'timeslots', [])) if hasattr(full_csp[x], 'timeslots') else 0
    )

def lcv_heuristic(csp, variable, assignment):
    values = sorted(
        csp[variable],
        key=lambda x: sum(len(csp[neighbor]) for neighbor in csp if x in csp[neighbor] and x not in assignment)
    )
    return values

def main():
    # Read dataset
    classes, teachers, rooms, timeslots, assignment = read_dataset("classdataset.txt")
    csp = {}

    # Measure execution time without heuristics
    start_time_without_heuristics = time.time()
    result_backtracking_without_heuristics = backtracking_search(csp, use_heuristics=False)
    end_time_without_heuristics = time.time()
    execution_time_without_heuristics = end_time_without_heuristics - start_time_without_heuristics

    # Measure execution time with heuristics
    start_time_with_heuristics = time.time()
    result_backtracking_with_heuristics = backtracking_search(csp)
    end_time_with_heuristics = time.time()
    execution_time_with_heuristics = end_time_with_heuristics - start_time_with_heuristics

    # Display the final schedule
    visualize_schedule(classes, assignment, teachers, rooms)

    # Plotting execution time comparison
    labels = ['Without Heuristics', 'With Heuristics']
    execution_times = [execution_time_without_heuristics, execution_time_with_heuristics]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, execution_times, color=['blue', 'orange'])
    plt.xlabel('Heuristics')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
