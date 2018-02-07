"""
When a beginner in a serious language asks "How do I blub?" they hear:

    "You probably shouldn't blub unless you know what you're doing..."
    "Haven't you studied blubology? Blub is far too complex to..."
    "Well it depends on the requirements from your Blub team..."
    "Just read all the manuals from the Blub developers and then..."

When a beginner in, say, PHP asks "How do I blub?" they hear:

    "Type in blub() and press enter"

This is why PHP, objectively one of the worst programming languages,
is still with us today. A great example of "Worse is Better"
"""
import sys

TIMEZONE = 'America/Los_Angeles'

class what_time_is_it(object):
    def __call__(self):
        import datetime
        import pytz
        tz = pytz.timezone(TIMEZONE)
        fmt = '%I:%M:%S %p, %a %b %m %Y'
        now = datetime.datetime.now(tz).strftime(fmt)
        return now

sys.modules[__name__] = what_time_is_it()
