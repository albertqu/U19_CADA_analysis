#author: Lexi Zhou

class Event_Node:
    def __init__(self):
        self.prev = None
        self.next = None

    def info(self):
        return self.__dict__


class BonsaiEvent(Event_Node):
    def __init__(self, event):
        self.item = event  # The bonsai_event object itself
        self.event_description = event[0]
        self.timestamp = event[1]
        self.event_code = event[2]
        self.restaurant = event[3]
        if len(event) > 4:
            self.keyword = event[4]


class Trial(Event_Node):
    """
    Series of bonsai events in the same restaurant stored as
    a doubly linked list
    """

    def __init__(self, first_event, list_of_bonsaievents, index):
        """
        events -- DLL: list of bonsai_event objects
        """
        self.enter = None
        self.firstEventNode = first_event
        self.tone_onset = None
        self.stimulation_on = None
        self.stimulation_off = None
        self.tone_prob = None
        self.restaurant = None
        self.T_Entry = None
        self.choice = None
        self.accept = 0
        self.outcome = None  #Quit, reward, or no reward
        self.reward = 0
        self.quit = None
        self.collection = None
        self.trial_end = None
        self.exit = None
        self.index = index
        self.lapIndex = 0
        self.blockIndex = 0
        self.item = list_of_bonsaievents
        self.comment = None

    def help(self):
        print('==enter==')
        print('timestamp for when animal enters a restaurant')
        print()
        print("==tone_onset==")
        print('timestamp for when offer tone starts playing')
        print()
        print("==stimulation==")
        print('0 as stimulation off and 1 otherwise')
        print()
        print('==T_Entry==')
        print('timestamp for sharp t junction entry')
        print()
        print('==exit==')
        print('Timestamp for the last event in this restaurant')
        print()
        print('==tone_prob==')
        print('Probability represented by the offer tone')
        print()
        print('==Restaurant==')
        print('Current Restaurant')
        print()
        print('==choice==')
        print('Given that a trial is initiated, this is the timestamp for either entering '
              'restaurant(accept) or exiting restaurant(reject')
        print()
        print('==accept==')
        print('boolean, whether it was an accept or not')
        print()
        print('==outcome==')
        print('timestamp for servo arm opem or no-reward tone played')
        print()
        print('==reward==')
        print('boolean for whether it was rewarded or not')
        print()
        print('==collection==')
        print('timestamp for when reward was taken, if this value is None, reward not taken')
        print()
        print('==quit==')
        print('timestamp for if a quit event happend')
        print()
        print('==lapIndex==')
        print('which trial it is in the current lap (1-2-3-4 count as one lap)')
        print()
        print('==blockIndex==')
        print('blocks of sequential trials, index for which block this trial is in')
        print()
        print('==comment==')
        print('Explanation for situations')
        print()
