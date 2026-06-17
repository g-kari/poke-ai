"""Day-1 #1 Crustle Wall agent — saved for benchmark comparison only.

Source: Kaggle notebook
  dashimaki360/beating-the-day-1-1-crustle-bot
  https://www.kaggle.com/code/dashimaki360/beating-the-day-1-1-crustle-bot

Author's TL;DR: simple rule-based scorer with action priority
ATTACH(1000) > EVOLVE(800) > PLAY(600) > ABILITY(400) > ATTACK(100) >
RETREAT(-1). The strength is the deck (pure Crustle wall + heal spam,
12 unique cards) which preys on the Pokemon-ex meta because of
Crustle's "Mysterious Rock Inn" ability blocking all damage from
opponent's ex Pokemon.

Adapted: replaced inline deck.csv path with RULE_DECK_PATH_CRUSTLE_DASHIMAKI
env-var override + fallback to deck_crustle_dashimaki.csv.

Original docstring follows.
"""

import os


import os
from cg.api import Observation, to_observation_class, OptionType, SelectContext, AreaType, Pokemon, Card

def read_deck_csv() -> list[int]:
    """Read this agent's dedicated deck."""
    file_path = os.environ.get("RULE_DECK_PATH_CRUSTLE_DASHIMAKI")
    if not file_path:
        here = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(os.path.dirname(here), "deck_crustle_dashimaki.csv")
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/deck.csv"
    with open(file_path) as file:
        csv = file.read().split(chr(10))
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    elif area == AreaType.HAND:
        return ps.hand[index]
    elif area == AreaType.DISCARD:
        return ps.discard[index]
    elif area == AreaType.ACTIVE:
        return ps.active[index]
    elif area == AreaType.BENCH:
        return ps.bench[index]
    elif area == AreaType.PRIZE:
        return ps.prize[index]
    elif area == AreaType.STADIUM:
        return obs.current.stadium[index]
    elif area == AreaType.LOOKING:
        return obs.current.looking[index]
    else:
        return None

def agent(obs_dict: dict) -> list[int]:
    """Pokémon Trading Card Game Agent.

    Rule:
    1. Perform preparation (Attach, Evolve, Play)
    2. Attack at the end
    3. Handle sub-selections (e.g., evolving via 'Kakusei' attack)
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        return read_deck_csv()

    select = obs.select
    options = select.option
    context = select.context

    scores = []
    for o in options:
        score = 0

        # 1. Main Turn Actions
        if context == SelectContext.MAIN:
            if o.type == OptionType.ATTACH:
                score = 1000
                # Attach "Hero's Cape" (ID: 1159) to Active Pokemon
                card = get_card(obs, o.area, o.index, obs.current.yourIndex)
                if card is not None and card.id == 1159:
                    if o.inPlayArea == AreaType.ACTIVE:
                        score = 2100
                    else:
                        # Do not attach to bench
                        score = 0
            elif o.type == OptionType.EVOLVE:
                score = 800
            elif o.type == OptionType.PLAY:
                score = 600
                card = get_card(obs, AreaType.HAND, o.index, obs.current.yourIndex)
                if card is not None:
                    # Use "Jumbo Ice" (ID: 1147) if Active Pokemon is damaged
                    if card.id == 1147:
                        active = obs.current.players[obs.current.yourIndex].active
                        if len(active) > 0 and active[0] is not None:
                            pokemon = active[0]
                            # Score highly if damaged and has 3+ energies
                            if pokemon.hp < pokemon.maxHp and len(pokemon.energies) >= 3:
                                score = 2000
                            else:
                                # Do not use if no damage or not enough energy
                                score = 0
                    # Use "Cook" (ID: 1212) if damaged
                    elif card.id == 1212:
                        active = obs.current.players[obs.current.yourIndex].active
                        if len(active) > 0 and active[0] is not None:
                            pokemon = active[0]
                            if pokemon.hp < pokemon.maxHp:
                                score = 1500
                            else:
                                score = 0
                    # Use "Cheren" (ID: 1224) to draw cards
                    elif card.id == 1224:
                        score = 1400
                    # Use "Battle Colosseum" (ID: 1264)
                    elif card.id == 1264:
                        score = 1300
            elif o.type == OptionType.ABILITY:
                score = 400
            elif o.type == OptionType.ATTACK:
                score = 100
            elif o.type == OptionType.RETREAT:
                score = -1

        # 2. Sub-selections (Context)
        else:
            # Base score for mandatory or context-specific choices
            score = 2000

            if o.type == OptionType.CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card != None:
                    # Logic for evolving via attack like 'Kakusei'
                    if context == SelectContext.EVOLVE or context == SelectContext.TO_BENCH:
                        # Higher score for Pokemon in deck/hand during search
                        score += 500

                    if isinstance(card, Pokemon):
                        # Targeted selection
                        if o.playerIndex != obs.current.yourIndex:
                            score += 500 if o.area == AreaType.ACTIVE else 100
                            score += len(card.energies) * 50
                        else:
                            score += card.hp

            elif o.type == OptionType.YES:
                score += 100
            elif o.type == OptionType.NUMBER:
                score += o.number

        scores.append(score)

    # Sort options by score descending
    sorted_options = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    output = []
    for i in range(min(len(sorted_options), select.maxCount)):
        idx = sorted_options[i]
        # Only include negative scores if we must meet minCount
        if scores[idx] >= 0 or len(output) < select.minCount:
            output.append(idx)

    return output
