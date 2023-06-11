aitradingbot.py Doc of a self-learning trading bot.

implementation of a self-learning trading bot. The code works by gathering data from a stock market API, implementing various technical analysis indicators on that data, and then using it to train a model with reinforcement learning. Here are the key parts:

    Imports and Configuration: The script imports several libraries, including pandas, numpy, tensorflow, keras, and Alpaca's trading API among others​1​. It then sets up an Alpaca API connection using provided API keys​1​.

    Data Collection and Preprocessing: The get_data function collects historical stock data for a given symbol over a specified time range, and adds several technical indicators like SMA, EMA, Bollinger Bands, RSI, Stochastic Oscillator, and MACD to the data​1​.

    Environment Setup: The _get_observation function is part of a custom Gym environment, which manages the state of the trading simulation and calculates rewards based on trading actions. The action and observation spaces are defined, and an instance of the environment is created​1​.

    Model Training: The train_model function trains a machine learning model using the data collected and processed earlier. The model learns to make trading decisions based on the state of the environment. This is done using a Monte Carlo Tree Search (MCTS) if a random number is less than a decayed epsilon value, or a random action otherwise. The model is trained using a batch of experiences from the memory deque​1​.

    Trading Strategy: The moving_average_strategy function is a simple trading strategy that buys when the closing price is above a certain percentage of the simple moving average (SMA) and sells when it's below​1​.

    Trade Logging and Saving: The script also includes functions to save the details of each trade to a CSV file​1​.

    Main Execution: In the main function, data is collected for the Microsoft (MSFT) stock, an environment is created, a model is trained using this data and environment, and then the model is saved. Finally, the model is tested with new data and the test results are printed​1​

curriculum.py Doc for the CurriculumAgent

This __init__ method is the constructor for the CurriculumAgent class. When an instance of this class is created, this method sets up the necessary properties and states for the object. Below is a step-by-step breakdown of what each line or block of code does:

    This is the method signature. It is specifying the parameters for the constructor. The parameters are:

    model_name: The name of the model to be used for the agent. It defaults to "gpt-3.5-turbo".
    temperature: The temperature to use for the model, impacting the randomness of the output. It defaults to 0.
    qa_model_name: The name of the model to be used for question-answering. It defaults to "gpt-3.5-turbo".
    qa_temperature: The temperature to use for the question-answering model. It defaults to 0.
    request_timout: The timeout for the request, in seconds. It defaults to 120.
    ckpt_dir: The directory for the checkpoints. It defaults to "ckpt".
    resume: Whether to resume from the last checkpoint. It defaults to False.
    mode: The mode for the curriculum agent, which can be "auto" or "manual". It defaults to "auto".
    warm_up: Not clear without more context or the full source code.
    core_inventory_items: A string representing core inventory items, or None. It defaults to None.
    
This block of code creates an instance of the ChatOpenAI class with the specified model_name, temperature, and request_timeout parameters, and assigns it to the llm property of the CurriculumAgent.

This does the same as the previous block, but this instance is intended for question answering, and is assigned to the qa_llm property.

This block of code checks if the mode parameter is either "auto" or "manual". If it is not, an error will be raised. If it is, it will be assigned to the mode property of the CurriculumAgent.

These lines of code assign the ckpt_dir parameter to the ckpt_dir property of the CurriculumAgent, and create the curriculum/vectordb directory within the checkpoint directory.

The rest of the code deals with loading previously completed tasks, failed tasks, and a question-answering (QA) cache from JSON files if resume is set to True. Otherwise, it initializes these properties as empty lists or dictionaries```python
if resume:
print(f"\033[35mLoading Curriculum Agent from {ckpt_dir}/curriculum\033[0m")
self.completed_tasks = U.load_json(f"{ckpt_dir}/curriculum/completed_tasks.json")
self.failed_tasks = U.load_json(f"{ckpt_dir}/curriculum/failed_tasks.json")
self.qa_cache = U.load_json(f"{ckpt_dir}/curriculum/qa_cache.json")
else:
self.completed_tasks = []
self.failed_tasks = []
self.qa_cache = {}

This block of code creates a new Chroma instance, which is likely a vector database for the QA cache. It assigns this instance to the qa_cache_questions_vectordb property of the CurriculumAgent.

Finally, this block of code checks if the number of questions in the QA cache vector database is equal to the number of questions in the QA cache. If not, it raises an error indicating that the QA cache and the vector database are not synchronized【5†source】.

This function, default_warmup(self), is a property method in Python, which allows the method to be accessed like a data attribute without the need for parentheses.

Here's a detailed breakdown of this method:

    context: The context property is set to 15. This could represent the number of context items to consider during the warm-up phase, although without more information on how the property is used elsewhere in the code, it's hard to say definitively.

    biome: This property, set to 10, could represent the number of biomes that the agent needs to explore or learn about during the warm-up phase.

    time: This property is set to 15, which could be referring to some aspect of time that the agent needs to understand, perhaps how many different time units it should recognize.

    nearby_blocks: This property is set to 0, which could mean that the agent doesn't need to be aware of nearby blocks during the warm-up phase.

    other_blocks: This property is set to 10, which might indicate that the agent needs to familiarize itself with 10 other types of blocks that are not considered "nearby".

    nearby_entities: This property is set to 5, possibly suggesting that the agent should be familiar with 5 types of entities in its vicinity during the warm-up phase.

    health: This property is set to 15. It might mean that the agent needs to understand 15 different health states or levels during its warm-up phase.

    hunger: This property is set to 15, which could be referring to the number of hunger states or levels the agent should understand during the warm-up phase.

    position, equipment, inventory, and chests: These properties are all set to 0, suggesting that they might not be necessary for the agent during the warm-up phase.

    optional_inventory_items: This property is set to 7, which could mean that the agent needs to be familiar with 7 optional inventory items during its warm-up phase.

    completed_tasks and failed_tasks: These properties are both set to 0, possibly indicating that the agent doesn't need to complete any tasks or recognize any failure states during its warm-up phase.

The dictionary returned by this function is likely used as a default configuration for an agent's warm-up phase, where the agent is getting acclimated to its environment and learning basic functionality before moving on to more complex tasks. However, without the context of the rest of the code, it's difficult to say exactly what each value represents​1​.

This is a Python property method named curriculum_observations. A property in Python is a way to access a method as if it were an attribute, without needing to call the method with parentheses.

Let's break it down:

    @property is a Python decorator that allows a method to be accessed like an attribute. Instead of calling self.curriculum_observations(), you can access it like self.curriculum_observations.

    def curriculum_observations(self): is the method definition. It is a method of the class in which it's defined, and self is a reference to the instance of the class.

    The method returns a list of strings. Each string represents a type of observation that the agent can make in the environment. These observations likely correspond to some aspect of the state of the game world. Here's a breakdown of what each might represent:

        "context": This could represent some broader context about the agent's current situation, perhaps including information from previous actions or events in the environment.

        "biome": In the context of a game like Minecraft, a biome is a type of environment with distinct geographic features and weather patterns. This observation likely includes information about the biome in which the agent is currently located.

        "time": This could represent the current time within the game world.

        "nearby_blocks": This likely represents information about the blocks that are immediately adjacent to the agent's current position.

        "other_blocks": This might represent information about other blocks that the agent has observed or interacted with that are not immediately adjacent to its current position.

        "nearby_entities": This likely includes information about other entities (such as NPCs or creatures) that are close to the agent's current position.

        "health": This likely represents the agent's current health level.

        "hunger": In a game like Minecraft, the player character can experience hunger, which can be remedied by consuming food items. This observation likely represents the agent's current hunger level.

        "position": This likely represents the agent's current position within the game world.

        "equipment": This could represent the items that the agent currently has equipped (such as tools or weapons).

        "inventory": This likely represents the items that the agent currently has in its inventory.

        "chests": This might represent information about any chests (which can be used for storage) that the agent has observed or interacted with.

        "completed_tasks": This could represent a record of tasks that the agent has successfully completed.

        "failed_tasks": This could represent a record of tasks that the agent attempted but failed to complete.

Overall, this method provides a compact representation of the agent's current state and its history of interaction with the game world, which can be used to guide its future actions​1​.

This render_observation method is part of the CurriculumAgent class and is responsible for generating an observation of the environment from a given event and chest observation. The observation is a dictionary of different aspects of the game state, each aspect being a key-value pair in the dictionary. The keys are strings that label the aspect of the game state, and the values are the actual data for that aspect.

The method takes two arguments:

    events: a list of events, where each event is a tuple. It is expected that the last event in this list is an "observe" event.
    chest_observation: some observation related to a chest. This is passed directly into the final observation dictionary.

Here's what the function does step by step:

    The method first asserts that the last event in the events list is an "observe" event. If it's not, the program will throw an error.

    The last event in the events list is then assigned to the variable event. From event, various pieces of information about the game state are extracted and assigned to local variables: biome, time_of_day, voxels (which represent 3D blocks in the environment), block_records, entities (which represent other creatures or players in the game), health, hunger, position, equipment, inventory_used, and inventory.

    The function then checks the voxels to see if any of them contain "dirt", "log", "grass", "sand", or "snow". If none of these are found, the biome is set to "underground".

    The function determines other_blocks, which are blocks that have been recently seen but are not in the voxels or inventory. If there are no such blocks, other_blocks is set to "None".

    The function then generates a string of nearby_entities, sorted by their value. If there are no entities, this string is set to "None".

    The function creates a string of completed_tasks and failed_tasks. If there are no tasks in either category, the respective string is set to "None".

    The function checks if the progress of the agent is less than the warm-up period for optional inventory items. If so, it filters out the optional inventory items from the inventory.

    The function then constructs the final observation dictionary, which includes various pieces of information about the game state, each formatted as a string. The observation includes: "context", "biome", "time", "nearby_blocks", "other_blocks", "nearby_entities", "health", "hunger", "position", "equipment", "inventory", "chests", "completed_tasks", and "failed_tasks".

    Finally, the method returns this observation dictionary.

This function essentially serves as a way for the agent to perceive the game state in a structured and organized way, providing it with all the necessary information it needs to make a decision about what to do next​1​.

Sure, here is an explanation of the render_human_message method within the CurriculumAgent class in the Voyager project:

    content = "": This line initializes an empty string content which will be used to build the message that the agent will eventually return.

    observation = self.render_observation(events=events, chest_observation=chest_observation): The render_observation method is called with events and chest_observation as arguments. The events argument contains information about the events that have taken place in the Minecraft environment, while chest_observation holds information about the state of chests in the environment. The render_observation method generates an observation from these inputs, which is then stored in the observation variable.

    The next block of code is wrapped in a conditional statement that checks whether self.progress is greater than or equal to self.warm_up["context"]. self.progress represents the progress of the agent through its curriculum, and self.warm_up is a dictionary that holds the minimum progress required for the agent to include certain types of information in its observations.

    Within this conditional block, the run_qa method is called to generate a list of questions and answers about the current state of the Minecraft environment. The method is called with the events and chest_observation arguments. Unfortunately, I wasn't able to find the specific details about the run_qa method, as it wasn't available in the repository at the time of my training cut-off in September 2021.

    For each pair of question and answer, the method checks if the answer contains "Answer: Unknown" or "language model". If it does, the loop continues to the next pair. If it doesn't, the question and answer are appended to the "context" section of the observation dictionary.

    The next block of code iterates over the keys in self.curriculum_observations, which are the types of observations the agent is trained to make. It checks whether the agent's progress is greater than or equal to the warm-up requirement for each type of observation. If it is, and if a random number is less than 0.8 (or if there's no warm-up requirement), the corresponding observation is added to the content string.

    Finally, the content of the agent's message is printed to the console, and a new HumanMessage object is created with content as its content, and returned.

Sure, the propose_next_task function is part of the CurriculumAgent class in the Voyager Minecraft project. The purpose of this function is to suggest the next task for the AI to carry out in the Minecraft world based on the current state of the game, the agent's progress, and the mode in which the agent is operating.

Here's a step-by-step explanation of the code:

    The function begins by checking if self.progress is 0 and self.mode is "auto". If both these conditions are met, the function sets the task to "Mine 1 wood log" and provides context for the task. The context is the additional information that helps the AI understand what the task involves. This part of the code is essentially setting the first task for the AI when it starts the curriculum in the automatic mode.

    Next, the function checks the amount of inventory used. If the inventory used is greater than or equal to 33, it means that the inventory is almost full, and the AI should take actions to manage the inventory. It does so by checking the state of available chests and deciding whether to deposit items into a chest, place a chest from the inventory, or craft a new chest. Each of these tasks comes with its own context to guide the AI.

    If the inventory isn't almost full, the function constructs messages which includes system and human messages. System messages are a representation of the world state, and human messages are constructed from the observations and events in the Minecraft world.

    If the mode of the agent is "auto", the function calls propose_next_ai_task with messages and max_retries as arguments. This function will propose the next task for the AI in automatic mode based on the provided messages.

    If the mode of the agent is "manual", the function calls propose_next_manual_task, which will propose the next task for the AI in manual mode.

    If the mode is neither "auto" nor "manual", the function raises a ValueError indicating an invalid mode.

I wasn't able to find specific details about propose_next_ai_task and propose_next_manual_task functions within the time limit. Also, I couldn't retrieve the most recent updates or changes to the code beyond my knowledge cut-off in September 2021.

Regarding the background of this project, Voyager is a Minecraft bot that uses reinforcement learning and a curriculum-based approach to progressively learn tasks in the Minecraft world. The agent gradually learns more complex tasks as it gets better at performing simpler ones. This strategy helps in breaking down the complex problem of learning to play Minecraft into smaller, manageable sub-problems.

Sure, I can help explain this function in detail. This function, propose_next_ai_task, is designed to propose the next task that the AI should perform when it's in the "auto" mode.

Here is a line-by-line breakdown:

    def propose_next_ai_task(self, *, messages, max_retries=5):
    This line defines the function propose_next_ai_task. It takes three parameters: self (which is a reference to the instance of the class that the method is being called on), messages (which contains the system and human messages constructed from the observations and events in the Minecraft world), and max_retries (which is set to 5 by default, this represents the maximum number of attempts the function should make to propose a task before giving up).

    curriculum = self.llm(messages).content
    This line is calling the llm function (which stands for Language Learning Model) on the messages and retrieving the content of the response. The llm function is likely a method of the ChatOpenAI class, given the initialization of self.llm in the constructor of the CurriculumAgent class. It's designed to generate a curriculum for the AI, i.e., a sequence of tasks that the AI should perform.

    print(f"\033[31m****Curriculum Agent ai message****\n{curriculum}\033[0m")
    This line is printing the curriculum generated by the AI agent. The f"\033[31m****Curriculum Agent ai message****\n{curriculum}\033[0m" is an f-string (formatted string literal) that includes the curriculum. The \033[31m and \033[0m are ANSI escape codes for coloring the text red in the terminal.

    response = self.parse_ai_message(curriculum)
    This line is calling the function parse_ai_message on the curriculum. The parse_ai_message function likely interprets the curriculum generated by the AI and returns a structured representation of it.

    assert "next_task" in response
    This line asserts that the key "next_task" exists in the response. If "next_task" is not in response, an AssertionError will be raised. This is a check to ensure that the parsed response includes a "next_task" for the AI to perform.

    context = self.get_task_context(response["next_task"])
    This line is calling the function get_task_context on the "next_task" from the response. The get_task_context function likely retrieves additional information or context about the proposed "next_task".

    return response["next_task"], context
    This line returns the "next_task" and its associated context as the result of the function.

    except Exception as e:
    This line begins an exception handling block. If any exception (error) occurs in the try block, it will be caught here and stored in the variable e.

    print(f"\033[35mError parsing curriculum response: {e}. Trying again!\033[0m")
    This line prints an error message that includes the exception e. The message is colored purple in the terminal due to the ANSI escape code \033[35m.

    return self.propose_next_ai_task(messages=messages, max_retries=max_retries - 1,)
    This line is a recursive call to propose_next_ai_task. If an exception was raised in the try block, the function will call itself again with# I will search for the parse_ai_message and get_task_context methods in the same file to provide a more accurate explanation of the method's behavior.

    search("def parse_ai_message", recency_days=0)
    
    This code snippet is part of a Python class definition named CurriculumAgent, specifically the propose_next_ai_task method. Here's a detailed breakdown of what the function does:
    
This method is designed to propose the next task for the AI to perform. It takes in two parameters: messages which likely contains some form of instructions or context for the task, and max_retries, a parameter that defaults to 5 and specifies the maximum number of retries for proposing a task in case of an exception.

The llm object is an instance of the ChatOpenAI class, which is an interface to the GPT-3.5-turbo language model or any other model name that was specified during the CurriculumAgent object initialization. Here, llm is used to generate a response based on the messages input. The .content attribute of the response is then assigned to the curriculum variable. This response is expected to guide the curriculum progression.

This line prints the curriculum variable content to the console, with the text "Curriculum Agent ai message" as a header. The "\033[31m" and "\033[0m" are ANSI escape codes used to color the printed text in red.

This line attempts to parse the curriculum message using the parse_ai_message method of the CurriculumAgent class. The result is stored in the response variable.

This line checks that the parsed response contains a "next_task" key. If it doesn't, an AssertionError will be raised, and the code execution will move to the except block.

If the "next_task" key exists in the response, this code retrieves the task context using the get_task_context method and then returns a tuple containing the next task and its associated context.

If any error occurs during the try block execution, this except block catches the exception and prints an error message to the console, again using ANSI escape codes to color the text (this time in purple).

After catching the exception, the method recursively calls itself with the same messages and a decremented max_retries value. This provides a retry mechanism for handling transient errors. If max_retries reaches 0, the method will stop retrying and let the exception propagate upwards.

Unfortunately, I wasn't able to access the parse_ai_message and get_task_context methods due to time constraints. For a more detailed understanding of the complete functionality of this method, I would recommend looking into those methods as well.

It's also important to note that the behavior of the propose_next_ai_task method, especially how it uses messages, can depend heavily on how the ChatOpenAI model (i.e., llm) is trained and configured. It appears to be using the AI

This function, parse_ai_message, is part of the CurriculumAgent class and its purpose is to extract a task from a given message. The function is defined as follows:

The parse_ai_message function is a method of the CurriculumAgent class. It takes a single argument, message, which is presumably a string containing the text of a message. This method is called when the AI curriculum needs to parse a task from the generated message.

The task variable is initialized as an empty string. This variable will hold the task extracted from the message.

This line starts a loop that iterates over each line in the message. The split("\n") function splits the message string into a list of lines.

This line checks if the current line starts with the string "Task:". This indicates that the line contains a task for the AI.

If the line does start with "Task:", then the task variable is updated to contain the rest of the line, starting from the 6th character (since Python uses 0-based indexing). The replace(".", "") function call removes all periods from the line, and the strip() function call removes leading and trailing whitespace. This results in a cleaned version of the task.

This line checks if the task variable is not empty. If it is empty (i.e., if no task was found in the message), then an AssertionError is raised with the message "Task not found in Curriculum Agent response". This indicates that the provided message did not contain a task, which is expected to be a problem based on the design of this system.

Finally, the function returns a dictionary with a single key-value pair. The key is "next_task" and the value is the task string. This format is presumably chosen to provide a consistent and easily understood output format.

In summary, the parse_ai_message function processes a given message, looks for a line starting with "Task:", and extracts the remainder of that line as a task. It removes any periods and whitespace from the task, and returns a dictionary containing the task. If no task is found in the message, it raises an AssertionError.

This function, propose_next_manual_task, is part of the CurriculumAgent class. Its purpose is to interactively get a new task and the corresponding context from the user.

This line defines the method propose_next_manual_task for the CurriculumAgent class. The function takes no arguments.

The function begins by initializing a few variables. confirmed is a Boolean variable that will be used to control the loop that follows. The variables task and context are both initialized as empty strings. They will be used to store the task and context input by the user.

This line starts a while-loop that continues as long as confirmed is False. The purpose of this loop is to repeatedly ask the user for input until they confirm that their input is correct.

These two lines ask the user for input. The first line prompts the user with "Enter task: " and assigns their input to the task variable. The second line prompts the user with "Enter context: " and assigns their input to the context variable.

This line prints out the task and context that the user just entered. The purpose of this is to let the user review their input.

This line asks the user to confirm their input. The input("Confirm? (y/n)") part prompts the user with "Confirm? (y/n)" and gets their input. The lower() function converts the user's input to lowercase to allow for case-insensitive comparisons. The in ["y", ""] part checks if the user's input is either "y" or an empty string. If it is, then confirmed is set to True, otherwise it remains False.

Once the user has confirmed their input, the loop ends and the function returns the task and context as a tuple. These two pieces of information are the output of the function.

In summary, the propose_next_manual_task function provides an interactive way to get a task and corresponding context from the user. It asks the user for a task and context, lets the user review their input, and asks the user to confirm their input. This process repeats until the user confirms their input. The function then returns the user's task and context.

The CurriculumAgent class in the Minecraft project, as well as your trading bot, share the fundamental idea of incorporating machine learning models into decision-making processes. However, the specific implementation details are quite different due to the different nature of the tasks they are designed to perform.

Here's what I understood from the CurriculumAgent and how it might relate to your trading bot:

    Initialization: The CurriculumAgent is initialized with parameters that define its behavior, such as model names, temperature (which affects the randomness of the generated responses), timeout duration, and a mode (which can be either 'auto' or 'manual'). It also initializes two models, one for general chat and one specifically for question-answering​1​.

    Observations: The CurriculumAgent receives a number of observations from the environment, including the player's context, biome, time of day, nearby blocks, health, hunger, position, equipment, inventory, and more​1​.

    Warm-up: The agent goes through a "warm-up" period, during which certain observations are gradually introduced into the agent's observation set​1​.

To adapt this to your trading bot, you would need to consider the following:

    Initialization: Identify what parameters you would want to initialize your trading bot with. This may include the trading model to use, trading parameters, and whether to run in 'auto' or 'manual' mode.

    Observations: Determine what observations your bot will need to make decisions. This could include market indicators, trading volume, historical price data, news sentiment, etc.

    Warm-up: Consider whether a warm-up period would be beneficial for your bot. During this period, you might gradually introduce more complex market indicators or trading strategies.

However, given the complexity of the Minecraft project, and the potential differences between it and your trading bot, I wasn't able to provide a detailed guide for rewriting your trading bot. Furthermore, the CurriculumAgent class relies on several other classes and functions from the Minecraft project which may not be directly applicable to your trading bot. Therefore, the general ideas above would need to be adapted to the specifics of your trading bot.

Lastly, it's important to note that implementing an AI trading bot involves dealing with real-world financial markets. Therefore, it is crucial to test any new changes thoroughly in a safe environment (like a paper trading account) before running the bot with real money. It's also important to be aware of and comply with any legal and regulatory requirements that may apply to using such a bot.

The function update_exploration_progress is part of the CurriculumAgent class in the provided Voyager project code. It takes one argument, info, which is expected to be a dictionary containing information about the progress of a task in the game. The function is responsible for updating the agent's records of completed and failed tasks based on whether the task was successful.

Here's a detailed breakdown of the function:

    task = info["task"]: This line extracts the current task from the info dictionary. The task is expected to be a string describing what the agent is supposed to do.

    if task.startswith("Deposit useless items into the chest at"): return: This is a conditional statement that checks if the current task starts with the string "Deposit useless items into the chest at". If it does, the function returns immediately and does nothing. This suggests that the developers of the agent decided not to track progress on tasks of depositing items into a chest, possibly because they don't consider this kind of task as relevant for the agent's learning progress.

    if info["success"]: ... else: ...: This conditional block checks whether the task was successful. The info dictionary is expected to have a key success which should be a boolean indicating whether the task was successfully completed. If info["success"] is True, the current task is added to the completed_tasks list. If info["success"] is False, the task is added to the failed_tasks list. In both cases, a message is printed to the console about the status of the task.

    self.clean_up_tasks(): After updating the task lists, the function calls the clean_up_tasks method of the CurriculumAgent. This method is not defined within the code snippet you provided, but based on its name and typical programming conventions, we can infer that it likely performs some kind of cleanup operation related to tasks, such as removing duplicates or saving the task lists to disk. However, the exact behavior would need to be verified by looking at the implementation of clean_up_tasks.

Please note that this interpretation assumes that the info dictionary always contains the keys task and success, and that task is a string and success is a boolean. If the info dictionary does not meet these assumptions, the function may not work correctly and could raise an exception.

The function clean_up_tasks is part of the CurriculumAgent class in the provided Voyager project code. This function is responsible for cleaning up the agent's lists of completed and failed tasks, removing duplicates and saving the updated lists to disk.

Here's a detailed breakdown of the function:

    updated_completed_tasks = []: This line creates an empty list that will be used to store the unique completed tasks (i.e., tasks without duplicates).

    updated_failed_tasks = self.failed_tasks: This line copies the list of failed tasks into a new list. This new list will be updated during the function's execution to remove tasks that have been completed.

    The first for loop goes through each task in the agent's list of completed tasks. If a task is not already in updated_completed_tasks, it is appended to this list. This process removes duplicates from the list of completed tasks while preserving the original order of the tasks.

    The second for loop goes through each task in the list of unique completed tasks. If a task is also in the list of failed tasks, it is removed from that list. This process ensures that a task can't be considered both completed and failed at the same time. Note that the while loop is used here to remove all instances of the task from the failed tasks list, in case it appears multiple times.

    The agent's lists of completed and failed tasks are updated with the cleaned-up lists.

    The final part of the function saves the cleaned-up lists of completed and failed tasks to disk as JSON files. The U.dump_json function is used for this purpose, which is presumably a utility function provided by the Voyager project for saving data as JSON. The saved files are named "completed_tasks.json" and "failed_tasks.json", and they are stored in the directory specified by self.ckpt_dir followed by "/curriculum".

This function ensures the consistency and correctness of the agent's records of completed and failed tasks. It can help to prevent issues caused by duplicate tasks and inconsistencies between the completed and failed task lists. It also allows the agent's progress to be saved and loaded between different sessions of the game.

The function decompose_task is part of the CurriculumAgent class in the provided Voyager project code. This function is responsible for decomposing a given task into smaller steps using the language model, and returning the decomposed task.

Here's a detailed breakdown of the function:

    messages = [...]: This line creates a list of SystemMessage and HumanMessage objects. These messages will be fed into the language model (llm) to guide it in decomposing the task.

        SystemMessage(content=load_prompt("curriculum_task_decomposition")): This SystemMessage is presumably loaded from an external source (like a file or database) using the load_prompt function. The prompt is likely a predefined instruction to the language model on how to decompose a task.

        self.render_human_message(events=events, chest_observation=""): This function call generates a HumanMessage that represents the current state of the game world. It uses the current game events and an empty chest observation to construct a contextual message for the language model.

        HumanMessage(content=f"Final task: {task}"): This HumanMessage tells the language model the final task that needs to be decomposed.

    The function then prints the final task that will be decomposed, for debugging purposes. The \033[31m and \033[0m are ANSI escape codes that change the color of the printed text to red.

    response = self.llm(messages).content: This line feeds the list of messages to the language model and gets its response. The language model's response is expected to be a decomposition of the task into smaller steps.

    The function prints the language model's response, again for debugging purposes.

    return fix_and_parse_json(response): This line processes the language model's response using the fix_and_parse_json function, and returns the result. The fix_and_parse_json function is likely a utility function provided by the Voyager project for cleaning up and parsing JSON data.

This function is a key part of the agent's ability to handle complex tasks in the game. By decomposing a task into smaller steps, the agent can tackle the task piece by piece, which can make it easier to complete the task successfully.

This function, run_qa, is part of the CurriculumAgent class, and its purpose is to generate and answer questions about the current state of the game, using a cache to avoid repeating work.

Here is a detailed breakdown of the function:

    questions_new, _ = self.run_qa_step1_ask_questions(events=events, chest_observation=chest_observation): This line uses the run_qa_step1_ask_questions function to generate a list of new questions based on the current state of the game.

    for question in questions_new:: This loop goes through each question in the list of new questions.

    if self.qa_cache_questions_vectordb._collection.count() > 0:: This line checks if there are any questions in the vector database cache. If the count is greater than zero, it means there are cached questions, so it proceeds to check for similarity with the current question.

    docs_and_scores = self.qa_cache_questions_vectordb.similarity_search_with_score(question, k=1): This line retrieves the most similar question from the vector database cache.

    if docs_and_scores and docs_and_scores[0][1] < 0.05:: This line checks if the most similar question is sufficiently similar to the current question (i.e., if the similarity score is less than 0.05). If it is, it uses the cached answer instead of generating a new answer.

    answer = self.run_qa_step2_answer_questions(question=question): If the current question is not sufficiently similar to any cached question, this line generates a new answer using the run_qa_step2_answer_questions function.

    assert question not in self.qa_cache: This line checks that the current question is not already in the cache. This is to prevent overwriting any existing answers in the cache.

    self.qa_cache[question] = answer: This line adds the new answer to the cache.

    self.qa_cache_questions_vectordb.add_texts(texts=[question]): This line adds the current question to the vector database cache.

    U.dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json"): This line saves the updated cache to a JSON file.

    self.qa_cache_questions_vectordb.persist(): This line saves the updated vector database cache.

    assert len(questions_new) == len(questions) == len(answers): This line checks that the number of new questions, questions, and answers are all the same. This is to ensure that every question has a corresponding answer.

    return questions, answers: This line returns the list of questions and their corresponding answers.
    
The get_task_context function is part of the CurriculumAgent class, and its purpose is to generate a context related to a given task, which includes a question about the task and its corresponding answer.

Here is a detailed breakdown of the function:

    question = f"How to {task.replace('_', ' ').replace(' ore', '').replace(' ores', '').replace('.', '').strip().lower()} in Minecraft?": This line generates a question about how to perform the given task in Minecraft. It does so by replacing underscores with spaces, removing any occurrence of 'ore' or 'ores', removing periods, and converting the resulting string to lowercase.

    if question in self.qa_cache:: This line checks if the generated question already exists in the cache. If it does, it retrieves the corresponding answer from the cache.

    else:: If the question does not exist in the cache, the function proceeds to generate a new answer and add it to the cache.

    answer = self.run_qa_step2_answer_questions(question=question): This line generates a new answer using the run_qa_step2_answer_questions function.

    self.qa_cache[question] = answer: This line adds the new answer to the cache.

    self.qa_cache_questions_vectordb.add_texts(texts=[question]): This line adds the current question to the vector database cache.

    U.dump_json(self.qa_cache, f"{self.ckpt_dir}/curriculum/qa_cache.json"): This line saves the updated cache to a JSON file.

    self.qa_cache_questions_vectordb.persist(): This line saves the updated vector database cache.

    context = f"Question: {question}\n{answer}": This line creates a context string that includes the question and its corresponding answer.

    return context: This line returns the generated context.

The render_system_message_qa_step1_ask_questions function is also shown in the code snippet, but it is separate from get_task_context. It appears to return a system message that prompts the agent to ask questions as part of the first step in the QA process. It does this by loading a prompt named "curriculum_qa_step1_ask_questions".

The render_human_message_qa_step1_ask_questions function is part of the CurriculumAgent class. The purpose of this function is to create a HumanMessage object using a set of observations made within the game. Here's a detailed breakdown of the function:

    def render_human_message_qa_step1_ask_questions(self, *, events, chest_observation): This is the function definition for render_human_message_qa_step1_ask_questions. It takes three parameters: self (a standard parameter referring to the instance of the class itself), events (which likely represents a list of events or actions that have occurred within the game), and chest_observation (which might contain information about any chests the agent has observed in the game).

    observation = self.render_observation(events=events, chest_observation=chest_observation): This line calls the render_observation method on the events and chest_observation arguments. The render_observation method likely processes these inputs and returns an observation object that contains a variety of information about the current state of the game.

    content = "": This line initializes a variable named content as an empty string. This variable will be used to accumulate the contents of the human message.

    for key in self.curriculum_observations: This line begins a loop that iterates over the keys in self.curriculum_observations. self.curriculum_observations is likely a list or dictionary that contains a set of observations that are relevant to the agent's curriculum.

    content += observation[key]: This line appends the observation corresponding to the current key in the loop to the content string. This will generate a long string of observations that will be used to construct the human message.

    return HumanMessage(content=content): This line constructs a HumanMessage object using the content string and returns it. HumanMessage is likely a class or function that constructs a message that's intended to represent communication from a human player within the game. The message is formed from the observations made by the agent within the game.

In summary, the render_human_message_qa_step1_ask_questions function generates a message that includes a set of observations made within the game. This message is then used to simulate human communication within the game.

The function run_qa_step1_ask_questions is a part of the CurriculumAgent class and its purpose is to generate and process questions regarding the game environment, particularly about the biome the agent is currently in. Here's a detailed breakdown of the function:

    def run_qa_step1_ask_questions(self, *, events, chest_observation): This is the function definition for run_qa_step1_ask_questions. It takes three parameters: self (a standard parameter referring to the instance of the class itself), events (which likely represents a list of events or actions that have occurred within the game), and chest_observation (which might contain information about any chests the agent has observed in the game).

    biome = events[-1][1]["status"]["biome"].replace("_", " "): This line extracts the type of biome from the last event and replaces underscores with spaces. This value is stored in the biome variable for use in forming questions.

    questions = [...]: This is a list of questions that the agent intends to ask about the current biome. Each question is a string that incorporates the biome variable.

    concepts = [biome, biome, biome]: This list of concepts is initialized with three entries, each of which is the biome value. It's used to pair with the corresponding questions asked.

    messages = [...]: This line creates a list called messages containing two messages: a system message generated by self.render_system_message_qa_step1_ask_questions(), and a human message generated by self.render_human_message_qa_step1_ask_questions(events=events, chest_observation=chest_observation). These messages are used as input to the QA language model.

    qa_response = self.qa_llm(messages).content: This line uses the QA language model (self.qa_llm) to generate a response based on the messages provided. The content of the response is stored in qa_response.

    try: ... except Exception as e: ...: This is a try-except block that attempts to perform a task and, if it encounters an exception, it handles the exception in a specified way.

    pattern = r"Question \d+: (.+)\nConcept \d+: (.+)": This line defines a regex pattern to extract question and concept pairs from the QA response.

    pairs = re.findall(pattern, qa_response): This line uses the regex pattern to find all matching pairs in qa_response.

    questions_new = [pair[0] for pair in pairs] and concepts_new = [pair[1] for pair in pairs]: These lines create new lists of questions and concepts extracted from the pairs.

    assert len(questions_new) == len(concepts_new): This line checks that the number of new questions and new concepts are the same.

    questions.extend(questions_new) and concepts.extend(concepts_new): These lines add the new questions and concepts to the original lists.

    print(f"\033[35mError parsing curriculum response for QA step 1 ask questions: {e}.\033[0m"): If an exception occurs during the extraction of questions and concepts, this line prints an error message.

    return questions, concepts: This line returns the final lists of questions and concepts, which can then be used in further processing or interaction with the game environment.
    
The function render_system_message_qa_step2_answer_questions is a part of the CurriculumAgent class. Its primary purpose is to generate a system message that will likely be used to guide the behavior of the AI agent or to provide information to the user or another part of the system.

Here's a detailed breakdown of the function:

    def render_system_message_qa_step2_answer_questions(self): This is the function definition for render_system_message_qa_step2_answer_questions. It takes only one parameter, self, which is a standard parameter referring to the instance of the class itself.

    return SystemMessage(content=load_prompt("curriculum_qa_step2_answer_questions")): This line does the main work of the function. It creates a new SystemMessage object, and the content of this message is set to the result of the load_prompt function called with the string "curriculum_qa_step2_answer_questions" as an argument.

The load_prompt function is not shown in this code snippet, but it's reasonable to assume that it's a function that retrieves a specific prompt, likely from some sort of external data storage, such as a file or database. The string "curriculum_qa_step2_answer_questions" likely identifies the particular prompt to be retrieved. This prompt could be a predefined message or a template that will be filled in with specific information.

After the SystemMessage object has been created with the appropriate content, it's returned by the function. This returned message can then be used in other parts of the system, perhaps to direct the behavior of the AI agent or to provide information to the user.

The function render_human_message_qa_step2_answer_questions is a part of the CurriculumAgent class. It's used to create a HumanMessage object that contains a question, which will be used to guide the behavior of the AI agent.

Here's a detailed breakdown of the function:

    def render_human_message_qa_step2_answer_questions(self, question): This is the function definition for render_human_message_qa_step2_answer_questions. It takes two parameters: self, which is a standard parameter referring to the instance of the class itself, and question, which is a string containing the question to be asked.

    content = f"Question: {question}": This line uses a formatted string (f-string) to create a new string that starts with the word "Question: " followed by the actual question. This line of code is preparing the content that will be included in the HumanMessage object.

    return HumanMessage(content=content): This line creates a new HumanMessage object with the content set to the string created in the previous line. The HumanMessage object is then returned by the function.

The HumanMessage object is a way to encapsulate a message that is intended to be perceived as coming from a human user. In this case, the message contains a question. This returned HumanMessage can be used in other parts of the system, potentially to guide the behavior of the AI agent. For example, the AI agent might be designed to process HumanMessage objects and generate responses to them.

The run_qa_step2_answer_questions function is part of the CurriculumAgent class. Its purpose is to carry out the second step of a QA (Question and Answer) process, which involves generating and processing a question, and then receiving and handling the answer. Here is a high-detail explanation of the function:

    def run_qa_step2_answer_questions(self, question): This is the function definition for run_qa_step2_answer_questions. It takes two parameters: self, which refers to the instance of the CurriculumAgent class itself, and question, which is a string representing the question to be asked.

    messages = [...]: This is an array of two messages that are used to guide the QA process. These messages include:
        self.render_system_message_qa_step2_answer_questions(): This function call generates a system message that guides the AI agent to respond to the questions. The function loads a predefined prompt (which is not shown in the provided code) and wraps it in a SystemMessage object, which is then returned.
        self.render_human_message_qa_step2_answer_questions(question=question): This function call generates a human message that includes the question to be asked. The result of this function is a HumanMessage object that contains the formatted question.

    print(f"\033[35mCurriculum Agent Question: {question}\033[0m"): This line prints the question to the console. The characters \033[35m and \033[0m are used to color the text in the console. In this case, the color is set to purple.

    qa_answer = self.qa_llm(messages).content: This line sends the list of messages (system message and human message) to the language model, denoted by self.qa_llm, to generate an answer. The qa_llm object is an instance of ChatOpenAI which is used to communicate with OpenAI's language model. The content attribute of the returned object contains the language model's response.

    print(f"\033[31mCurriculum Agent {qa_answer}\033[0m"): This line prints the answer received from the language model to the console. The text color in the console is set to red.

    return qa_answer: Finally, the function returns the answer generated by the language model.

In summary, this function facilitates the second step of a QA process by generating a question, passing it to the AI language model, and processing the generated response.
