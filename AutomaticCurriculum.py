class AutomaticCurriculum:
    def __init__(self, initial_tasks, bot):
        self.tasks = initial_tasks
        self.current_task_index = 0
        self.bot = bot

    def get_next_task(self):
        # If all tasks have been completed, generate new tasks
        if self.current_task_index >= len(self.tasks):
            self.generate_new_tasks()
            self.current_task_index = 0

        task = self.tasks[self.current_task_index]
        self.current_task_index += 1
        return task

    def generate_new_tasks(self):
        # Get the current skills of the bot
        current_skills = self.bot.get_current_skills()

        # Get the current state of the market
        market_state = self.bot.get_market_state()

        # Generate new tasks based on the current skills and market state
        if "skill1" in current_skills:
            self.tasks.append("task related to skill1")
        if "skill2" in current_skills and market_state == "state1":
            self.tasks.append("task related to skill2 and state1")
        # ... add more conditions as needed
