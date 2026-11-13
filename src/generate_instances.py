import json
import random
import os

building= ["TFPETR", "HOLDEN", "BIOL", "ELECE", "SCIENC"]
capacity= [40,50,60,80,100]

metadata= [{"scale": "small","seed": 42,"num_courses": 25,"num_rooms": 6,"num_timeslots": 8},
{"scale": "medium","seed": 42,"num_courses": 50,"num_rooms": 12,"num_timeslots": 16},
{"scale": "large","seed": 42,"num_courses": 100,"num_rooms": 20,"num_timeslots": 24}]


for x in metadata:
  for y in range(5):
    currentseed = x["seed"]+y
    random.seed(currentseed)

    num_rooms=x["num_rooms"]
    num_timeslots= x["num_timeslots"]
    num_courses= x["num_courses"]
    
    rooms = []
    for r in range(num_rooms):
        rooms.append({
            "id": f"R{r}",
            "building": random.choice(building),
            "capacity": random.choice(capacity)
        })

    timeslots = []
    slotsperday = num_timeslots // 2
    for t in range(num_timeslots):
        day_group = "MWF" if t < slotsperday else "TR"
        slot_order = (t % slotsperday) + 1
        timeslots.append({
            "id": t,
            "day_group": day_group,
            "slot_order": slot_order,
            "is_undesirable": (slot_order == 1 or slot_order == slotsperday)  # First and last slot of the day
        })

    num_cohorts = max(4, num_courses // 6)
    num_instructors = max(4, num_courses // 3)
    courses = []
    for c in range(num_courses):
        chosen_room = random.choice(rooms)
        courses.append({
            "id": f"CS_{1000 + c}",
            "instructor_id": f"I{c % num_instructors}",
            "enrollment": random.randint(15, chosen_room["capacity"]),
            "student_cohort": f"C{c % num_cohorts}",
            "preferred_slots": random.sample(range(num_timeslots), k=2),
            "preferred_days": [random.choice(["MWF", "TR"])]
        })

    instance_data = {
      "metadata": {
        "scale": x["scale"],
        "seed": currentseed,
        "num_courses": num_courses,
        "num_rooms": num_rooms,
        "num_timeslots": num_timeslots
      },
      "rooms": rooms,
      "timeslots": timeslots,
      "courses": courses
    }
    filepath=f"data/{x["scale"]}_seed{currentseed}.json"
    with open(filepath, "w") as f:
        json.dump(instance_data, f, indent=2)


print("Data generated.")