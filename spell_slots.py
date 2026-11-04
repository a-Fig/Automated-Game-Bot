import threading

SLOT_COUNT = 6  # Q W E R D F, left to right on the action bar


class FakeObject:
    # stands in for a slot the spell model returned nothing for
    def __init__(self, class_name, x=0):
        self.class_name = class_name
        self.x = x
        self.y = 0
        self.confidence = 0


class SpellSlotMapper:
    """
    Puts spell detections into fixed action bar slots.

    Sorting the detections by x and padding the end breaks when the model misses a
    slot in the middle: every spell to its right slides down one index. Instead,
    whenever the model returns exactly SLOT_COUNT detections, their x centres are
    recorded as the slot positions. Any other frame snaps each detection to the
    nearest recorded slot, and slots with no detection are treated as cooldown.
    Until the first full frame arrives, every slot reads as cooldown, so the bot
    holds its spells instead of casting from a guessed layout.
    """

    def __init__(self, slot_count=SLOT_COUNT):
        self.slot_count = slot_count
        self.slot_x = None
        self.lock = threading.Lock()

    def is_calibrated(self):
        with self.lock:
            return self.slot_x is not None

    def map(self, predictions):
        predictions = sorted(predictions, key=lambda p: p.x)

        if len(predictions) == self.slot_count:
            with self.lock:
                self.slot_x = [p.x for p in predictions]
            return predictions

        with self.lock:
            slot_x = self.slot_x

        if slot_x is None:
            return [FakeObject("cooldown") for _ in range(self.slot_count)]

        result = [FakeObject("cooldown", x) for x in slot_x]
        best_confidence = [-1.0] * self.slot_count
        for p in predictions:
            i = min(range(self.slot_count), key=lambda s: abs(slot_x[s] - p.x))
            # two boxes on one slot: keep the more confident one
            if p.confidence > best_confidence[i]:
                result[i] = p
                best_confidence[i] = p.confidence
        return result
