# Yingying Wang
# 1127423
# yingyw@cs.washington.edu
# CSE473 HW2


# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        DisToCloestGhost = newFood.height+newFood.width
        for successor in successorGameState.getGhostStates():
          DisToCloestGhost = min(DisToCloestGhost, manhattanDistance(newPos, successor.getPosition()))
        foods = newFood.asList()
        closestFood = 0
        if foods:
          foodDistances = []
          for food in foods:
            foodDistances.append(manhattanDistance(newPos, food))
          closestFood = min(foodDistances)
        if DisToCloestGhost < 3:
          return DisToCloestGhost*100000
        else:
          return (newFood.height*newFood.width-len(foods))*100000+((newFood.height+newFood.width)-closestFood)*1000

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
    def isTerminal(self, state, depth):
        if state.isWin() or state.isLose() or self.depth == depth:
          return True
        return False

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        max_value = -float('inf')
        values = []
        newAction = None
        for action in actions:
            value = self.min_val(gameState.generateSuccessor(0, action),1,0)
            values.append(value)
            if value == max(values):
              newAction = action
        return newAction
    def max_val(self, gameState, move, depth):
      if self.isTerminal(gameState, depth):
        return scoreEvaluationFunction(gameState)
      agentsNumber = gameState.getNumAgents()
      agent = move % agentsNumber
      actions = gameState.getLegalActions(0)
      result = []
      for action in actions:
        successor = gameState.generateSuccessor(agent, action)
        result.append(self.min_val(successor, move+1, depth))
      return max(result) 
    def min_val(self, gameState, move,depth):
      if self.isTerminal(gameState, depth):
        return scoreEvaluationFunction(gameState)
      agentsNumber = gameState.getNumAgents()
      agent = move % agentsNumber
      actions = gameState.getLegalActions(agent)
      result = [] 
      if move == agentsNumber-1:
        result = [self.max_val(gameState.generateSuccessor(agent, action), 0, depth+1) for action in actions]
        return min(result)
      else:
        result = [self.min_val(gameState.generateSuccessor(agent, action), move+1, depth) for action in actions]
        return min(result)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_val(self, gameState, move,depth, x, y):
      if self.isTerminal(gameState, depth):
        return scoreEvaluationFunction(gameState)
      agentsNumber = gameState.getNumAgents()
      agent = move % agentsNumber
      actions = gameState.getLegalActions(0)
      result = []
      for action in actions:
        successor = gameState.generateSuccessor(agent, action)
        result.append(self.min_val(successor, move+1, depth,x,y))
        value = max(result)
        if value>y:
          return value
        x = max(x,value)
      return max(result) 
    def min_val(self, gameState, move, depth,x, y):
      if self.isTerminal(gameState, depth):
        return scoreEvaluationFunction(gameState)
      agentsNumber = gameState.getNumAgents()
      agent = move % agentsNumber
      actions = gameState.getLegalActions(agent)
      result = [] 
      if move == agentsNumber-1:
        for action in actions:
          value = self.max_val(gameState.generateSuccessor(agent, action), 0, depth+1,x,y)
          result.append(value)
          value = min(result)
          if value<x:
            return value
          y = min(y, value) 
      else:
        for action in actions:
          value = self.min_val(gameState.generateSuccessor(agent, action), move+1, depth,x,y)
          result.append(value)
          value = min(result)
          if value < x:
            return value
          y = min(y, value)
      return min(result) 
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        x = -float('inf')
        y = float('inf')
        actions = gameState.getLegalActions(0)
        max_value = -float('inf')
        values = []
        newAction = None
        for action in actions:
            value = self.min_val(gameState.generateSuccessor(0, action),1,0,x,y)
            values.append(value)
            if value == max(values):
              newAction = action
            if value > y:
              return value
            x = max(x,value)
        return newAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        result = None
        max_val = -float('inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
          value = self.get_Value(gameState.generateSuccessor(0, action),1)
          if value > max_val:
            max_val = value
            result = action
        return result

    def get_Value(self, gameState, move):
      agentsNum = gameState.getNumAgents()
      agent = move % agentsNum
      depth = move / agentsNum
      if (self.isTerminal(gameState, depth)):
        return self.evaluationFunction(gameState)
      max_val = -float('inf')
      actions = gameState.getLegalActions(agent)
      sums = 0.0
      if agent:
        for action in actions:
          sums+=self.get_Value(gameState.generateSuccessor(agent, action), move + 1)
        return sums/len(actions)
      else:
        for action in actions:
          max_val =max(self.get_Value(gameState.generateSuccessor(agent, action), move + 1), max_val)
        return max_val


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    result = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()    
    minDisCapsules = float("inf")
    if len(capsules) == 0:
      minDisCapsules = 0
    else:
      for capsule in capsules:
        minDisCapsules = min(minDisCapsules, manhattanDistance(capsule, newPos))
    result += 1.0/(1+minDisCapsules)

    minDisFood = float("inf")
    foods = currentGameState.getFood().asList()
    if len(foods)==0:
      minDisFood = 0
    else:
      for food in foods:
        minDisFood = min(minDisFood, manhattanDistance(newPos, food))
    result += 1.0/(1+minDisFood)
    newGhostStates = currentGameState.getGhostStates()
    result += sum([ghostState.scaredTimer for ghostState in newGhostStates])
    return result
# Abbreviation
better = betterEvaluationFunction

