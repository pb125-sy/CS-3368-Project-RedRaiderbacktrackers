import json
import random

#loads data from filepath to problem_data dictionary
def load_instance(filepath: str) -> dict:
    with open(filepath, "r") as f:
        fileload = json.load(f)
    room_dict = {x["id"]: x for x in fileload["rooms"]}
    timeslot_dict = {x["id"]: x for x in fileload["timeslots"]}
    course_dict = {x["id"]: x for x in fileload["courses"]}
    return {
        "metadata": fileload.get("metadata", {}),
        "rooms": room_dict,
        "timeslots": timeslot_dict,
        "courses": course_dict,
    }

#creates initial random schedule assignment dictionary following format {course_id: (room_id, timeslot_id)} for each element
def create_initial_state(problem_data: dict, seed: int=42) -> "TimetableState":
    random.seed(seed)
    assignment = {}
    timeslot_ids = list(problem_data["timeslots"].keys())

    for course_id, course_info in problem_data["courses"].items():
        valid_rooms = [
            r_id
            for r_id, r_info in problem_data["rooms"].items()
            if r_info["capacity"] >= course_info["enrollment"]
        ]
        assignment[course_id] = (random.choice(valid_rooms), random.choice(timeslot_ids))

    return TimetableState(problem_data, assignment)


class TimetableState:
    def __init__(self, problem_data: dict, assignments: dict[str, tuple[str, int]]):
        self.problem = problem_data
        self.assignments = assignments.copy()

    def copy(self) -> "TimetableState":
        return TimetableState(self.problem, self.assignments)
    
    def evaluate(self, hardweight: int=10000) -> tuple [int, int, int]:
        hard_count=0
        soft_cost=0
        courses = self.problem["courses"]
        rooms = self.problem["rooms"]
        timeslots = self.problem["timeslots"]

        course_ids = list(self.assignments.keys())

        #check pairwise hard constraints
        for i in range(len(course_ids)):
            c1 = course_ids[i]
            r1, t1 = self.assignments[c1]
            for j in range(i + 1, len(course_ids)):
                c2 = course_ids[j]
                r2, t2 = self.assignments[c2]

                if t1 == t2:
                    # Room overlap
                    if r1 == r2:
                        hard_count += 1
                    # Instructor clash
                    if courses[c1]["instructor_id"] == courses[c2]["instructor_id"]:
                        hard_count += 1
                    # Student cohort overlap
                    if courses[c1]["student_cohort"] == courses[c2]["student_cohort"]:
                        hard_count += 1

        #per-course hard capacity and individual soft preferences
        for c_id, (r_id, t_id) in self.assignments.items():
            course = courses[c_id]
            if rooms[r_id]["capacity"] < course["enrollment"]:
                hard_count += 1

            if t_id not in course["preferred_slots"]:
                soft_cost += 5
            if timeslots[t_id]["day_group"] not in course["preferred_days"]:
                soft_cost += 4
            if timeslots[t_id]["is_undesirable"]:
                soft_cost += 3

        #group soft constraints: student schedule gaps per day
        cohorts = {c["student_cohort"] for c in courses.values()}
        for cohort in cohorts:
            for day in ("MWF", "TR"):
                orders = [
                    timeslots[t_id]["slot_order"]
                    for c_id, (_, t_id) in self.assignments.items()
                    if courses[c_id]["student_cohort"] == cohort
                    and timeslots[t_id]["day_group"] == day
                ]
                orders.sort()
                for idx in range(len(orders) - 1):
                    if orders[idx + 1] - orders[idx] > 1:
                        soft_cost += 5

        #group soft constraints: instructor classes in different buildings per day
        instructors = {c["instructor_id"] for c in courses.values()}
        for inst in instructors:
            for day in ("MWF", "TR"):
                bldgs = {
                    rooms[r_id]["building"]
                    for c_id, (r_id, t_id) in self.assignments.items()
                    if courses[c_id]["instructor_id"] == inst
                    and timeslots[t_id]["day_group"] == day
                }
                if len(bldgs) > 1:
                    soft_cost += 4
            
        total_cost=(hard_count*hardweight)+soft_cost
        return (total_cost, hard_count, soft_cost)
    
    #swap heler function
    def swap(self, course_a: str, course_b: str):
        self.assignments[course_a], self.assignments[course_b] = (
            self.assignments[course_b],
            self.assignments[course_a],
        )

    #only tabu search needs first value in this returned tuple
    #generates all valid neighbor states, returning a list of tuples of (move information, neigbor state), for each neighbor.
    def get_neighbors(self) -> list[tuple[tuple, "TimetableState"]]:
        rooms = self.problem["rooms"]
        timeslots = self.problem["timeslots"]
        courses = self.problem["courses"]
        neighbors = []
        occupied_slots = set(self.assignments.values())

        #valid reassign moves
        for currcourse, curr_assign in self.assignments.items():
            enrollment = courses[currcourse]["enrollment"]
            for r_id, r_info in rooms.items():
                if r_info["capacity"] < enrollment:
                    continue
                for t_id in timeslots:
                    if (r_id, t_id) != curr_assign and (r_id, t_id) not in occupied_slots:
                        move = ("reassign", currcourse, curr_assign[0], curr_assign[1])
                        next_state = self.copy()
                        next_state.assignments[currcourse] = (r_id, t_id)
                        neighbors.append((move, next_state))

        #valid swap moves
        course_ids = list(self.assignments.keys())
        for i in range(len(course_ids)):
            c_a = course_ids[i]
            r_a, _ = self.assignments[c_a]
            for j in range(i + 1, len(course_ids)):
                c_b = course_ids[j]
                r_b, _ = self.assignments[c_b]

                if (
                    rooms[r_b]["capacity"] >= courses[c_a]["enrollment"]
                    and rooms[r_a]["capacity"] >= courses[c_b]["enrollment"]
                ):
                    move = ("swap", c_a, c_b)
                    next_state = self.copy()
                    next_state.swap(c_a, c_b)
                    neighbors.append((move, next_state))

        return neighbors