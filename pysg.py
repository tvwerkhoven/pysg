#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
#
# pysg - process whatsapp log files into IRC-like statistics page, like pisg
# 
## Todo:
# Add 'happy/sad' -- check which are happy and sad smileys - MEDIUM -- DONE partially
#

from IPython import embed

import json
import unittest
import os
import time
import numpy as np
import random
import datetime
import pandas as pd
import re
import operator
import argparse
import logging
from collections import defaultdict
import yaml

# Need more than this amount of messages to count as monologue.
MONOLOGUE_THRESHOLD = 5
# Number of letters for a word to be a long word
LONGWORD_THRESHOLD = 6

# Loosely based on https://www.wikiwand.com/en/List_of_emoticons#/Western
re_smiley_happy = re.compile("[xX:;8=]'?[\‑\-o^]?[\]\)3>}DPp]")
re_smiley_sad = re.compile(">?[:;]'?[\‑\-]?[\(\[@<]")

# From https://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python#26568779
re_emoji_smiley_happy = re.compile(u'['
    u'\U0001F600-\U0001F60F'
    u'\U0001F617-\U0001F61D]+?',
    re.UNICODE)

re_emoji_smiley_sad = re.compile(u'['
    u'\U0001F610-\U0001F616'
    u'\U0001F61E-\U0001F62D]+?',
    re.UNICODE)

def load_config(cfgfile):
    """
    Load use alias config so we split main program and chat-specific settings
    """

    # Init with default values
    config = {'user_aliases': None}

    try:
        with open(cfgfile) as fd:
            config = yaml.safe_load(fd)
    except FileNotFoundError as e:
        raise FileNotFoundError("Could not find configuration file: " + str(e))

    return config

def parse_log(chatlog, useraliases=None, parsedlogfile=None, timeframes=(31,), chatstatsfile=None, template=None):
    """
    Parse chat log files from 'chatlog', export standardized CSV to 'parsedlogfile',
    parse chat statistics over period of 'timeframes' days (iterable)
    """
    # Input file, should be whatsapp format (although format changes all the 
    # time)
    timestr = time.strftime("%Y%m%d_%H%M%S",time.localtime())
    outdir = "pysg_{}".format(timestr)
    os.makedirs(outdir)

    # Make full path if not None
    if (parsedlogfile):
        parsedlogfile = os.path.join(outdir, parsedlogfile)
    if (chatstatsfile):
        chatstatsfile = os.path.join(outdir, chatstatsfile)

    # Read and normalize log file, optionally store normalized file
    chatnormalized = normalize_whatsapp(chatlog, parsedlogfile)

    dedup_usernames(chatnormalized, useraliases)

    dfchat = mk_dataframe(chatnormalized)

    # Calculate statistics & optionally store as pickle & json
    allstats = calc_stats_per_tf(dfchat, timeframes, useraliases)
    if (chatstatsfile): store_stats(allstats, chatstatsfile)

    # Publish results
    publish(allstats, outdir, template, "pysg_stats.html")
    
def normalize_whatsapp(chatlog, parsedlogfile=None):
    """
    Given a raw log file from WhatsApp, normalize into parsable chatlog

    Oldest format: (not supported anymore)
    17.12.11, 14:45:42: User Name: Test?
    Late 2013 format: hours are not zero-padded!
    17/12/2011, 4:45:42: User Name: Test?
    2019+ format:
    [27/11/2013, 04:49:52] User Name: Bluf dat alles in t russisch is

    """

    # Pre-compile regexp to check if line is a new message or not.
    # Multi-lines messages span multiple log lines. If a line does 
    # not start with a date, we assume it's part of the previous line
    # WhatsApp format 2013 and newer:
    re_checknewline = re.compile("([0-9/]{10}\, [0-9]+:[0-9]+:[0-9]+)")
    # re_checknewline.search("[03/03/2019, 19:57:26] nick name: Tof")
    # re_checknewline.search("17/12/2011, 14:45:42: User Name: Test?")

    # Parse data, concatenate messages when spanning multiple lines.
    chatnormalized = []
    parsedmsg = ""

    # @TODO Fix this ugly code duplication
    if (parsedlogfile):
        with open(chatlog) as f, open(parsedlogfile, 'w') as w:
            # Read and parse line immediately
            # https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list
            for r in f:
                # We keep trailing whitespace and port it to the parsed file
                # r.rstrip()
                # Whatsapp sometimes inserts \u200e characters, strip these here
                r = r.replace("\u200e","")

                # If line starts with date, it's a new message
                date_re = re_checknewline.search(r)
                if (date_re):
                    # We found a new message, write out previous unles empty
                    if (parsedmsg): 
                        w.write(",".join(parsedmsg))
                        chatnormalized.append(parsedmsg)
                    # Start parsing new message, could be multiple lines. Use 
                    # the regexp from above to delineate the date start and 
                    # end, as well as the username start
                    parsedmsg = normalize_whatsapp_line(r, dstart=date_re.start(), dend=date_re.end(), ustart=date_re.end()+2)
                else:
                    # This line continues the message on the previous line
                    parsedmsg[-1] = parsedmsg[-1].rstrip() + r
    else:
        with open(chatlog) as f:
            # Read lines only
            # https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list
            for r in f:
                # We keep trailing whitespace and port it to the parsed file
                # r.rstrip()
                # Whatsapp sometimes inserts \u200e characters, strip these here
                r = r.replace("\u200e","")

                # If line starts with date, it's a new message
                date_re = re_checknewline.search(r)
                if (date_re):
                    # We found a new message, write out previous unles empty
                    if (parsedmsg): 
                        chatnormalized.append(parsedmsg)
                    # Start parsing new message, could be multiple lines
                    parsedmsg = normalize_whatsapp_line(r, dstart=date_re.start(), dend=date_re.end(), ustart=date_re.end()+2)
                else:
                    # This line continues the message on the previous line
                    parsedmsg[-1] = parsedmsg[-1].rstrip() + r

    return chatnormalized

def normalize_whatsapp_line(r, dstart=1, dend=21, ustart=23):
    """
    Parse single whatsapp line into normalized log file. Expects clean string,
    i.e. without U200e characters. 
    
    dstart: date string start
    dend: date string end
    ustart: username start
    """

    if (r.find(" changed the subject to")>dend):
        # Line is new topic
        # example subject line: 19/05/2023, 11:34:34: Nick Name changed the subject to “🐮🌻🍺#!bla🍺🌻🐮”
        mtype = "subject"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" changed the subject to",ustart)]
        msgstr = r[r.find("changed the subject to",dend):]
    elif (r.find(" created")>dend and r.find(":",ustart) == -1):
        # Line is created message, no colon found
        # example icon line: 
        mtype = "create"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" created",ustart)]
        msgstr = r[r.find("created",dend):]
    # Whatsapp <2014
    elif (r.find(" changed the group icon")>dend):
        mtype = "icon"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" changed the group icon",ustart)]
        msgstr = r[r.find("changed the group icon",dend):]
    elif (r.find(" deleted the group icon")>dend):
        mtype = "icon"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" deleted the group icon",ustart)]
        msgstr = r[r.find("deleted the group icon",dend):]

    elif (r.find(" changed this group's icon")>dend):
        # Line is new topic
        # example icon line: 27/04/2016, 20:14:33: Nick Extra Name changed this group's icon
        mtype = "icon"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" changed this group's icon",ustart)]
        msgstr = r[r.find("changed this group's icon",dend):]
    elif (r.find(" deleted this group's icon")>dend):
        # Line is new topic
        # example icon line: 27/04/2016, 20:14:33: Nick Extra Name deleted this group's icon
        mtype = "icon"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" deleted this group's icon",ustart)]
        msgstr = r[r.find("deleted this group's icon",dend):]
    elif (r.find(" added")>dend and r.find(":",ustart) == -1):
        # Line is add message, no colon found
        # example icon line: 
        mtype = "join"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" added",ustart)]
        msgstr = r[r.find("added",dend):]
    elif (r.find(" joined")>dend and r.find(":",ustart) == -1):
        # Line is leave message, no colon found
        # example icon line: 
        mtype = "join"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" joined",ustart)]
        msgstr = r[r.find("joined",dend):]
    elif (r.find(" left")>dend and r.find(":",ustart) == -1):
        # Line is leave message, no colon found
        # example icon line: 
        mtype = "leave"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(" left",ustart)]
        msgstr = r[r.find("left",dend):]
    elif (r.find("'s security code changed")>dend):
        # Security code changed
        # example: 23/09/2016, 22:24:16: Nick Extra Name's security code changed.
        mtype = "seccode"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find("'s security code changed",dend):]
    elif (r.find("Messages to this group are now secured with end-to-end encryption.")>dend):
        mtype = "secured"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find("Messages to this group are now secured with end-to-end encryption.",dend):]
    elif (r.find("image omitted")>dend):
        # Image posted - without caption
        # example: 30/04/2017, 12:29:02: Nick Extra Name: image omitted
        # NB we strip \u200e above, so no need to check
        mtype = "image"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find(":",ustart)+2:]
    elif (r.find("GIF omitted")>dend):
        # GIF posted - without caption
        mtype = "gif"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find(":",ustart)+2:]
    elif (r.find("video omitted")>dend):
        # GIF posted - without caption
        mtype = "video"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find(":",ustart)+2:]
    elif (r.find("audio omitted")>dend):
        # GIF posted - without caption
        mtype = "audio"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find(":",ustart)+2:]
    elif (r.find("sticker omitted")>dend):
        # GIF posted - without caption
        mtype = "sticker"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find(":",ustart)+2:]
    elif (r.find("document omitted")>dend):
        # GIF posted - without caption
        mtype = "document"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find(":",ustart)+2:]
    else:
        # Line contains regular message
        # split lines by date [dstart:dend], then until the next colon, then until the end
        # example line:         27/11/2019, 21:40:56: Nick Extra Name: Zo, even whatsapp geleegd\n
        mtype = "message"
        datestr = r[dstart:dend]
        userstr = r[ustart:r.find(":",ustart)]
        msgstr = r[r.find(":",ustart)+2:]

    # Userstr can contain unicode stuff (u202a, u202c, xa0 xa0) if it's a 
    # phone number, filter that out
    # \u202a+32\xa07\xa012345678\u202c
    userstr = userstr.strip('\u202a\u202c').replace(u'\xa0', u' ')

    return [datestr, userstr, mtype, msgstr]
        
def mk_dataframe(chatparsed):
    # Convert data to Pandas Dataframe for easier sorting later on
    dfchat = pd.DataFrame(chatparsed, columns=("date", "user", "mtype", "content"))
    # We use exact date matching instead of to_datetime() to speed up processing
    # 15/05/2016, 21:55:11
    logging.debug("Parsing dates...")
    # dfchat['date'] = dfchat.apply(lambda row: pd.to_datetime(row['date'], format="%d/%m/%Y, %H:%M:%S", exact=True, cache=True), axis=1)
    # datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

    # Time can be 1:12:12 or 12:12:12, so be clever
    dfchat['date'] = dfchat.apply(lambda v: datetime.datetime(
        int(v['date'][6:10]), # Year
        int(v['date'][3:5]), # Month
        int(v['date'][0:2]), # Day
        int(v['date'][12:-6]), # H -- can be 1 or 2 digits, but always starts at letter 12 and ends -6
        int(v['date'][-5:-3]), # M 
        int(v['date'][-2:])), # S
        axis=1)

    dfchat.set_index('date', inplace=True)
    logging.debug("parsing dates complete...")

    # Calculate number of words per message, smileys, caps, etc.
    # https://stackoverflow.com/questions/26568722/remove-unicode-emoji-using-re-in-python#26568779
    re_emoji = re.compile(u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]+', 
        re.UNICODE)
    
    dfchat['words'] = dfchat.apply(lambda row: len(row["content"].split()), axis=1)

    return dfchat

def dedup_usernames(chatparsed, useraliases):
    """
    Using a list of aliases, replace all aliases with the primary name
    """

    if (useraliases == None):
        return

    # Make lookup dict from alias list.
    # Input: useraliases: {<name> : [alias, alias, alias], <name>: [alias, alias], ...}
    # Output: userdict: {<alias>: name, <alias>: name, ...}
    userdict = {}
    for primname, aliases in useraliases.items():
        for a in aliases:
            userdict[a] = primname

    # For each name, lookup primary name
    for i, row in enumerate(chatparsed):
        # Look up name in dict, if doesn't exist, return name itself
        # (i.e. don't change) as fallback
        primname = userdict.get(row[1], row[1])
        # Update name inplace
        chatparsed[i][1] = primname

def calc_stats_emoji(dfchat):
    """
    Calculate: 
    1. emoji use per user

    output:

    emoji1
        User1: count1
        User2: count2
        UserN: countN
    emojiN
        User2: count2

    etc.
    """

    # Use defaultdict to count frequency
    # https://stackoverflow.com/questions/893417/item-frequency-count-in-python
    emoji_all = defaultdict(int)
    emoji_user = {}

    # Messages per user, used to normalize emoji counts
    msg_per_user = dfchat.groupby('user').count()['content']

    # Loop over all messages and users, search for emoji, count and store
    for msgstr, userstr in zip(dfchat['content'], dfchat['user']):
        r_hap = re_smiley_happy.findall(msgstr) + re_emoji_smiley_happy.findall(msgstr)
        r_sad = re_smiley_sad.findall(msgstr) + re_emoji_smiley_sad.findall(msgstr)
        r_emoji = r_hap + r_sad

        for e in r_emoji:
            emoji_all[e] += 1
            if (not emoji_user.get(e)):
                emoji_user[e] = defaultdict(int)
            emoji_user[e][userstr] += 1
    
    # # Prepare output format. Index 0 is for all users, other keys are per user
    # emoji_topx = {}
    # emoji_topx[0] = sorted(emoji_all.items(), key=operator.itemgetter(1), reverse=True)[:20]
    # for u, e in emoji_user.items():
    #     emoji_topx[u] = sorted(e.items(), key=operator.itemgetter(1), reverse=True)[:5]
    
    # users = emoji_topx.keys()

    return emoji_user

def calc_monologues(dfchat):
    """
    Check:
    - longest monologues per user
    - count of monologues per user
    """

    # Keep track of monologue count per user
    monologues = defaultdict(dict)
    randmonologues = defaultdict(list)

    lastuser = None
    monocounter = 0

    # The code below is 1000x faster than this: for i, m in dfchat.iterrows(): msgstr = m['content']
    for msgstr, userstr in zip(dfchat['content'], dfchat['user']):
        if userstr == lastuser:
            monocounter += 1
            thismonologue.append(msgstr)
        else:
            if (monocounter > MONOLOGUE_THRESHOLD):
                if (monocounter > monologues[lastuser].get('longest',{'length':0})['length']):
                    monologues[lastuser]['longest'] = {
                        'length': len(thismonologue),
                        'message': thismonologue
                    }
                monologues[lastuser]['count'] = monologues[lastuser].get('count', 0) + 1
                randmonologues[lastuser].append(thismonologue)

            monocounter = 0
            thismonologue = [msgstr]
        lastuser = userstr
    
    # Select random monologues
    for user in monologues.keys():
        monologues[user]['random'] = random.choice(randmonologues[user])

    return monologues

def calc_mostactive(dfchat):
    """
    Calculate:
    1. most active nick all time, by number of messages
    2. most active nick by time of day, by number of messages

    return:
    dict with keys 'allday', 0, 6, 12, 18
    and list of values: (nick, messages, words)
    """

    # This is possible and elegant, but creates a multi-index dataframe which
    # makes indexing more complex
    # https://stackoverflow.com/questions/14529838/apply-multiple-functions-to-multiple-groupby-columns
    # def func_random_msg(x):
    #     return x[np.random.randint(len(x))]
    # func_random_msg.__name__ = 'random'
    # dfchat.groupby('user', sort=False).agg(
    #     {'content':[len, func_random_msg],
    #     'words': np.average}
    #     )

    # Most active nick by time of day
    active = {}

    for hr in ('allday', 0, 6, 12, 18):
        if (hr == 'allday'):
            dfchatsub = dfchat
            dictkey = hr
        else:
            msk = (dfchat.index.hour >= hr) & (dfchat.index.hour <  (hr+6))
            dfchatsub = dfchat[msk]
            dictkey = "{:02d}:00".format(hr)
        
        # Get: random quote, count of messages (using len), average words per
        # message. We hijack the 'mtype' column here to compute the message count
        # so we don't need a multi-index thing (see above)
        activenick = dfchatsub.groupby('user', sort=False).agg(
            {'content': lambda x: x[np.random.randint(len(x))],
            'mtype': len,
            'words': lambda x: round(np.average(x),2) }
            ).sort_values('mtype', ascending=False)

        # Rename columns to understand meaning
        # https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas/46912050#46912050
        activenick.columns = 'random', 'messages', 'wordsperline'

        active[dictkey] = activenick.transpose().to_dict()

    # For allday stats, calculate how active each nick is per time of day, as
    # percentage of all messages
    for user in active['allday'].keys():
        maxval = 0
        maxidx = '00:00'
        totalactivity = 0
        for hr in (0, 6, 12, 18):
            # User might not have activity in a time of day, use get(). User
            # entry *must* exist in 'allday' bc we loop over it

            # TODO BUG this rounding is usually OK, but off by one or two is 
            # possible (e.g. .4 .4 .4 .8 rounds to 1 instead of 2)
            # This problem is known as the 
            # https://en.wikipedia.org/wiki/Proportional_representation 
            # and https://en.wikipedia.org/wiki/Single_transferable_vote
            # https://stackoverflow.com/questions/16226991/allocate-an-array-of-integers-proportionally-compensating-for-rounding-errors/20054616
            #
            # Poor man's solution:
            # l=[.4, .4, .4, .8]

            # l0 = [round(i*3/(sum(l))) for i in l] # NOK
            # # Correct delta
            # delta = 3 - sum(l0)
            # # change largest element because: 
            # # - it cannot be larger than total because the largest element cannot be larger than the sum
            # # - it cannot be smaller than 0 because ...
            # l0[l0.index(max(l0))] += delta

            thisidx = "{:02d}:00".format(hr)
            activityhr = round(100*active["{:02d}:00".format(hr)].get(user, {'messages':0})['messages']/active['allday'][user]['messages'])
            totalactivity += activityhr
            if (activityhr > maxval):
                maxval = activityhr
                maxidx = thisidx

            active['allday'][user][thisidx] = activityhr
        deltafix = 100 - totalactivity
        active['allday'][user][maxidx] += deltafix

    return active

def calc_alltime_stats(dfchatsub):
    """
    Calculate # of messages per day during night, morning, afternoon, 
    evening. Always group data per day, aggregate again in plotting when
    necessary

    Group data per day for <50 days, by week for <365 days, by month for all else
    """

    td = (dfchatsub.index.max()  - dfchatsub.index.min()).days
    if (td > 1500):
        tfreq = 'M'
    elif (td > 50):
        tfreq = 'W'
    else:
        tfreq = 'D'
    # tfreq = 'D'

    dfcont = dfchatsub['content']
    dfdate = pd.date_range(dfcont.index.min().date(), dfcont.index.max(), freq=tfreq)

    # Count messages for 4 quadrants of the day. reindex to force same date range, fillna() for missing days
    msg0 = dfcont[(dfcont.index.hour >= 0) & (dfcont.index.hour < 6)].resample(tfreq).count().reindex(dfdate).fillna(0)
    msg6 = dfcont[(dfcont.index.hour >= 6) & (dfcont.index.hour < 12)].resample(tfreq).count().reindex(dfdate).fillna(0)
    msg12 = dfcont[(dfcont.index.hour >= 12) & (dfcont.index.hour < 18)].resample(tfreq).count().reindex(dfdate).fillna(0)
    msg18 = dfcont[(dfcont.index.hour >= 18) & (dfcont.index.hour < 24)].resample(tfreq).count().reindex(dfdate).fillna(0)

    # Convert to python datatypes, we don't need numpy here. Also convert 
    # dates to days and drop midnight timestamp
    alltime = {}
    for d, msgs in zip(dfdate, zip(msg0.values, msg6.values, msg12.values, msg18.values)):
        alltime[str(d.date())] = {
            '00:00': int(msgs[0]),
            '06:00': int(msgs[1]),
            '12:00': int(msgs[2]),
            '18:00': int(msgs[3]),
            'all': int(sum(msgs))
        }

    return alltime

def calc_daily_stats(dfchatsub):
    """
    Calculate relative number of messages and words per hour of day
    """

    # relative messages and # words per hour of day, in percentage
    dfdaily_contents = dfchatsub['content'].groupby(lambda x: x.hour).count()
    dfdaily_words = dfchatsub['words'].groupby(lambda x: x.hour).sum()
    # words per message (or line) per hour of day
    dfdaily_wordsperline = dfchatsub['words'].groupby(lambda x: x.hour).sum() / dfchatsub['content'].groupby(lambda x: x.hour).count()

    # Force display of each hour, also if no data (use 0) -- not sure why I made this int64
    hridx = np.arange(24, dtype=np.int64)
    dfdaily_contents = dfdaily_contents.reindex(hridx).fillna(0)
    dfdaily_words = dfdaily_words.reindex(hridx).fillna(0)
    dfdaily_wordsperline = dfdaily_wordsperline.reindex(hridx).fillna(0)

    # TODO: can be more elegant, construct directly from iterables?
    daily = {}
    for hr, contents, words, wordsperline in zip(hridx, dfdaily_contents, dfdaily_words, dfdaily_wordsperline):
        daily["{:02d}:00".format(hr)] = {
            'messages': contents,
            'words': words,
            'wordsperline': wordsperline
        }

    return daily

def calc_total_stats(dfchat):
    """
    Calculate totals of chat data:
    1. Last topic
    2. Number of messages
    3. Number of words
    4. Number of images
    5. Number of icons
    """

    totals = {}
    # If no last topic, use last message
    totals['lasttopic'] = dfchat[dfchat['mtype']=='message']['content'][-1]
    try:
        totals['lasttopic'] = dfchat[dfchat['mtype']=='subject']['content'][-1]
    except:
        pass

    totals['date'] = {
        'oldest': dfchat.index.min().ctime(),
        'oldestmsg': dfchat['content'][0],
        'newest': dfchat.index.max().ctime(),
        'newestmsg': dfchat['content'][-1],
        'duration': int((dfchat.index.max() - dfchat.index.min()).days)
    }
    totals['words'] = int(dfchat['words'].sum())
    totals['images'] = int(sum(dfchat['mtype']=='image'))
    totals['messages'] = int(sum(dfchat['mtype']=='message'))
    totals['icons'] = int(sum(dfchat['mtype']=='icon'))
    totals['users'] = {
        'count': int(len(dfchat['user'].unique())),
        'names': dfchat['user'].unique().tolist()
        }
    totals['maxwords'] = {
        'count': int(dfchat['words'].max()), 
        'user': dfchat[dfchat['words'] == dfchat['words'].max()]['user'][0],
        'message': dfchat[dfchat['words'] == dfchat['words'].max()]['content'][0]
    }

    return totals

def calc_word_analysis(dfchat):
    """
    Analyse words/messages
    1. top 15 most used long words (longer than LONGWORD_THRESHOLD) per user
    2. Amount of screaming (!) or asking (?) per user
    3. Amount of CAPS per user
    """

    # Optional: Don't include nicknames as most used words: flatten list of 
    # names, count number of names, use this as minimum top x+10
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists#45323085
    # import functools
    # user_nicks = functools.reduce(operator.iconcat, USER_ALIASES, [])
    # num_words = len(user_nicks) + 20

    num_words = 15

    # Split all messages and make lowercase for easier comparison. 
    # Convert to series so we can use value_counts() later on
    dfchat_msg = dfchat[dfchat['mtype'] == 'message']

    # This code calculates all words, not per user:

    # # dfwords = pd.Series(' '.join(dfchat_msg['content']).lower().split())
    # # TODO Issue: this also breaks links / URLs bc . is removed
    # dfwords = pd.Series(' '.join(dfchat_msg['content']).replace("!","").replace("?","").lower().split())

    # # https://stackoverflow.com/questions/3939361/remove-specific-characters-from-a-string-in-python
    # # %timeit w=' '.join(dfchat_msg['content']).lower().split() - 0.5ms -- reference time, the rest of the code should be fast compared to this existing code
    # # %timeit w.translate({ord(i):None for i in '!?.,'}) - 2.7ms
    # # %timeit w.replace("!","").replace("?","").replace(",","").replace(".","") - 0.15ms
    # # %timeit "".join(list(filter(lambda ch: ch not in " ?!.,", w))) - 5ms

    longmost_u_out = defaultdict(dict)
    shout_u_out = {}
    ask_u_out = {}
    caps_u_out = {}

    for u in dfchat_msg['user'].unique():
        # Get list of words for this user, remove (some) punctuation
        dfchat_msg_u = dfchat_msg[(dfchat_msg['user'] == u)]

        # Relative # of caps messages
        caps_u_out[u] = round(sum(1 for x in dfchat_msg_u['content'] if x.isupper()) * 100.0 / len(dfchat_msg_u),3)

        # We need this operation in the end:
        #   dfwords_u = pd.Series(' '.join(dfchat_msg_u['content']).replace("!","").replace("?","").lower().split())
        # but by sequencing it we can get ? and ! counts per user
        dfwords_u = ' '.join(dfchat_msg_u['content'])
        shout_u_out[u] = len(dfwords_u)
        dfwords_u = dfwords_u.replace("!","")
        shout_u_out[u] -= len(dfwords_u)

        ask_u_out[u] = len(dfwords_u)
        dfwords_u = dfwords_u.replace("?","")
        ask_u_out[u] -= len(dfwords_u)

        # Normalize
        ask_u_out[u] /= round(float(len(dfwords_u))/100.0,3)
        shout_u_out[u] /= round(float(len(dfwords_u))/100.0,3)

        # for date, l in dfchat.iterrows():
        #     print(l['content'])
        #     len(list(filter(str.isupper, l)))
        #     break

        dfwords_u = pd.Series(dfwords_u.lower().split())

        # Count long words, sort, limit to top-X
        longmost_u = dfwords_u[dfwords_u.str.len() > LONGWORD_THRESHOLD].value_counts(sort=True)[:num_words]

        # Make dict of words and users, i.e. 'word': {'user1': count1, 'userN': countN}
        # Note that not all users are included in all words. tolist() to get 
        # ints that work in JSON
        for w, c in zip(longmost_u.index, longmost_u.values.tolist()):
            longmost_u_out[w][u] = c

    return caps_u_out, shout_u_out, ask_u_out, longmost_u_out

def calc_timing(dfchat):
    """
    Calculate timing between messages and of conversations, in two regimes:
    1. During a conversation (messages follow <2min)
    2. Outside conversation -- deprecated, boring regime

    Output:

    [no]chat:
        bin:
            user: count
            user: count
            user: count
        bin:
            user: counts
    """

    timing_hist_users = []
    timing_hist_bins = None
    timing_hist_counts = []

    for u in ['Everyone'] + dfchat['user'].unique().tolist():
        if (u == 'Everyone'):
            dfchat_sub = dfchat
        else:
            dfchat_sub = dfchat[dfchat['user'] == u]
            # Only calculate if we have more than 100 messages for this user
            if len(dfchat_sub) < 100:
                continue
        
        timing_hist_users.append(u)

        # Calculate gaps between times
        timing = (dfchat_sub.index[1:] - dfchat_sub.index[:-1]).total_seconds()

        # Get 99%-max value:
        # timing_99max = np.percentile(timing, 99)
        # Use fixed max timing instead to ensure same bins across users
        timing_99max = 48*3600

        # Chat timing in seconds, from 0 to 60. Optionally use density=True, 
        # but this fails work when no data is present
        timing_hist_chat = np.histogram(timing, bins=30, range=(0, 60))
        # Only story bins the first time, should not be necessary as they're all equal
        if (u == 'Everyone'):
            timing_hist_bins = timing_hist_chat[1][:-1].tolist()
        timing_hist_counts.append(timing_hist_chat[0]*100)

        # Nochat timing in minutes, from 1 min to 99% value
        # timing_hist_nochat = np.histogram(timing/60., bins=30, range=(1, 1+timing_99max/60.))
    
    # Construct dict here, with layout as specified above
    chat_timing = {}
    for bin_i, bin_val in enumerate(timing_hist_bins):
        chat_timing[bin_val] = {}
        for u, c in zip(timing_hist_users, timing_hist_counts):
            chat_timing[bin_val][u] = round(float(c[bin_i]),3)

    return {'chat': chat_timing}

def calc_network(dfchat, useraliases):
    """
    Calculate social network:
    1. For each user, who do they mention and how much?
    Output:
    From, to, weight
    """

    if (useraliases == None):
        return {
            'total': 0,
            'network': 0
        }

    # Most mentioned nicknames
    # https://stackoverflow.com/questions/13062402/find-the-index-of-the-column-in-data-frame-that-contains-the-string-as-value
    # https://stackoverflow.com/questions/43289269/pandas-equivalent-for-grep
    # https://www.highcharts.com/demo/dependency-wheel
    mentioned_network = defaultdict(dict)
    mentioned_total = {}

    aliasdict = useraliases    
    # For each user (object), count how many times they are mentioned by others (subject)
    for u_object in dfchat['user'].unique():
        # Loop over aliases for object to get all aliases for object
        aliases_object = aliasdict.get(u_object, []) + [u_object]

        # For each alias of this user (object)
        for alias_object in aliases_object:
            # Count by whom (subject) this alias (object) is mentioned
            mask_mentioned = dfchat['content'].str.lower().str.contains(alias_object.lower(), regex=False)
            counts_per_subject = dfchat['user'][mask_mentioned].value_counts()
            
            # Store subjects who mentioned this alias
            for u_subject, count in counts_per_subject.iteritems():
                mentioned_network[u_subject][u_object] = mentioned_network[u_subject].get(u_object, 0) + count
            # Store total mentions for this object
            mentioned_total[u_object] = mentioned_total.get(u_object, 0) + int(counts_per_subject.sum())

    # mentioned_total_srt = sorted(mentioned_total.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'total': mentioned_total,
        'network': mentioned_network
    }

def calc_lonely(dfchat):
    """
    Calculate which messages are least interacted with (long silence before 
    AND after). Also count messages that are first and last (silence longer 
    than 6 hrs).

    output:
    lonely:
        date1:
            user: username
            message: message
            gapbefore: silence before this message, hrs
            gapafter: silence after this message, hrs
        date2:
            ...
        dateN:
            ...
    
    first/last:
        user1: count
        user2: count
        ...
    """
    # Calculate timing between messages (only)
    dfchat_sub = dfchat[dfchat['mtype'] == 'message']
    timing = (dfchat_sub.index[1:] - dfchat_sub.index[:-1]).total_seconds()

    # Calculate loneliest messages, i.e. ones that have the longest silence 
    # before *and* after them. We take the minimum of the silence before and 
    # after each message as metric. Summing them would incorrectly judge the 
    # last message of a conversation as 'lonely'
    message_gap = np.vstack([timing[:-1], timing[1:]]).min(axis=0)

    # Sort and find top 20 loneliest messages (= maximum gap)
    lonely_msg_idx = message_gap.argsort()[-20:][::-1]

    # Story in dict
    lonely = {}
    for i in lonely_msg_idx.tolist():
        lonely[dfchat_sub.index[i+1].ctime()] = {
            'user': dfchat_sub.user[i+1],
            'message': dfchat_sub.content[i+1],
            'gapbefore': round(timing[i]/3600.,2),
            'gapafter': round(timing[i]/3600.,2)
        }

    # Calculate last word (more than 6 hrs no message)
    last_msg_idx = np.argwhere(timing > 3600*6).flatten()
    last = defaultdict(dict)
    first = defaultdict(dict)
    for i in last_msg_idx.tolist():
        last[dfchat_sub.iloc[i]['user']]['count'] = last[dfchat_sub.iloc[i]['user']].get('count',0) + 1
        last[dfchat_sub.iloc[i]['user']]['random'] = last[dfchat_sub.iloc[i]['user']].get('random',[]) + [dfchat_sub.iloc[i]['content']]

        first[dfchat_sub.iloc[i+1]['user']]['count'] = first[dfchat_sub.iloc[i+1]['user']].get('count',0) + 1
        first[dfchat_sub.iloc[i+1]['user']]['random'] = first[dfchat_sub.iloc[i+1]['user']].get('random',[]) + [dfchat_sub.iloc[i+1]['content']]

    for k in last.keys():
        last[k]['random'] = random.choice(last[k]['random'])
    for k in first.keys():
        first[k]['random'] = random.choice(first[k]['random'])

    return lonely, first, last

def calc_stats(dfchat, useraliases):
    """
    Given normalized log format in chatparsed, calculate chat statistics.
    """

    emoji = calc_stats_emoji(dfchat)

    monologues = calc_monologues(dfchat)

    lonely, first, last = calc_lonely(dfchat)

    activenick = calc_mostactive(dfchat)

    capsing, shouting, asking, mostwords = calc_word_analysis(dfchat)

    alltimestats = calc_alltime_stats(dfchat)

    dailystats = calc_daily_stats(dfchat)

    totalstats = calc_total_stats(dfchat)

    timing = calc_timing(dfchat)

    network = calc_network(dfchat, useraliases)

    return {'emoji': emoji, 
        'monologues': monologues, 
        'lonely': lonely,
        'first': first,
        'last': last,
        'active': activenick, 
        'mostwords': mostwords, 
        'capsing': capsing,
        'shouting': shouting,
        'asking': asking,
        'alltime': alltimestats,
        'daily': dailystats,
        'totals': totalstats,
        'timing': timing,
        'network': network
        }

def store_stats(chatstats, chatstatsfile):
    pd.to_pickle(chatstats, chatstatsfile + ".pickle")
    #Could also work, might be the same: pickle.dump(chatstats, "dump2.pickle")
    import json
    
    # Json serialization only works if we don't have DataFrame or NumPy stuff
    # in our output. The below loop helps to debug which element might fail
    # as it will raise an error locally.
    try:
        json.dumps(chatstats)
    except:
        for k0,v0 in chatstats.items():
            for k, v in v0.items():
                json.dumps(v)

    with open(chatstatsfile + ".json", 'w') as fd:
        json.dump(chatstats, fd, indent=1)

def calc_stats_per_tf(dfchat, timeframes, useraliases):
    """
    Calculate chat statistics for a number of timeframes in days.

    dfchat should be a pandas dataframe with date as index.
    timeframes should be an iterable with time durations in days
    """
    allstats = {}
    last_tf_real = None

    # First convert -1 timeframe to 100 years.
    timeframes = list(timeframes)
    if (-1 in timeframes):
        timeframes[timeframes.index(-1)] = 100*365

    for tf in sorted(timeframes):
        # Filter dataframe on timeframe, select last tf days of data
        date_min = dfchat.index.max() - pd.Timedelta(tf, unit='d')
        dfchatsub = dfchat[dfchat.index > date_min]

        # Calculate actual timespan we're parsing
        tf_real = (dfchatsub.index.max()  - dfchatsub.index.min()).days
        # If we don't have more data for this timeframe, than the previous skip it
        if (tf_real == last_tf_real): continue

        # Calculate statistics, store thi
        allstats[tf_real] = calc_stats(dfchatsub, useraliases)
        last_tf_real = tf_real

    return allstats

def mk_html_report(allstats, outpath, template):
    """
    Given dict of statistics 'allstat', make HTML report out using 'template' as starting point.

    Layout is as follows:

    - all-time graph
    - time of day graph
    - top posters all time
    - top posters per time of day
    """
    return

def prep_render(s0):
    """
    Given dict of statistics, prep render for HTML output
    """
    render = {}
    render['daily'] = ''
    for k,v in s0['daily'].items():
        render['daily'] += "{},{:.0f},{:.1f}\n".format(k,v['messages'], v['wordsperline'])

    render['alltime'] = ''
    for k,v in s0['alltime'].items():
        render['alltime'] += "{},{:d},{:d},{:d},{:d}\n".format(k,v['00:00'], v['06:00'], v['12:00'], v['18:00'])

    render['totals'] = s0['totals']
    render['lonely'] = s0['lonely']
    render['active'] = s0['active']['allday']


    # In: dict[user][count/random]
    # out: [user, count, random]
    render['first'] = sorted([[u, v['count'], v['random']] for u, v in s0['first'].items()], key=lambda x: x[1], reverse=True)
    render['last'] = sorted([[u, v['count'], v['random']] for u, v in s0['last'].items()], key=lambda x: x[1], reverse=True)

    # Sort monologues
    topmon = []
    for user,v in s0['monologues'].items():
        topmon.append([v['count'], user, v['longest']['length'], v['random']])
    topmons = sorted(topmon,reverse=True)
    render['monologues'] = topmons

    ### MOST EMOJIS USED (same as most words)

    # Build matrix containing: [word, countuser1, countuser2, countuser3, countuserN]
    # Problem: list of words are different per user
    emoji_mat = np.array([list(s0['emoji'].keys())],dtype=object).T
    allusers = []
    # Loop over all words, then loop over all users that mentioned this word, 
    # store in big matrix. Each time we find a new user, expand matrix accordingly
    for emoji, emojihits in s0['emoji'].items():
        for user, count in emojihits.items():
            if (not user in allusers):
                allusers.append(user)
                emoji_mat = np.hstack([emoji_mat, np.zeros((len(emoji_mat), 1),dtype=int)])
            emoji_mat[emoji_mat[:,0] == emoji,allusers.index(user)+1] = count
    
    # Sort matrix, sum only user counts
    emoji_mats = emoji_mat[emoji_mat[:,1:].sum(1).argsort()[::-1]][:15]

    render['emoji'] = ",".join(['Word'] + allusers)
    for words in emoji_mats:
        render['emoji'] += "\n" + ','.join([str(i) for i in words])

    ### MOST WORDS USED

    # Build matrix containing: [word, countuser1, countuser2, countuser3, countuserN]
    # Problem: list of words are different per user
    longmost_mat = np.array([list(s0['mostwords'].keys())],dtype=object).T
    allusers = []
    # Loop over all words, then loop over all users that mentioned this word, 
    # store in big matrix. Each time we find a new user, expand matrix accordingly
    for word, wordhits in s0['mostwords'].items():
        for user, count in wordhits.items():
            if (not user in allusers):
                allusers.append(user)
                longmost_mat = np.hstack([longmost_mat, np.zeros((len(longmost_mat), 1),dtype=int)])
            longmost_mat[longmost_mat[:,0] == word,allusers.index(user)+1] = count
    
    # Sort matrix, sum only user counts
    longmost_mats = longmost_mat[longmost_mat[:,1:].sum(1).argsort()[::-1]][:15]

    render['mostwords'] = ",".join(['Word'] + allusers)
    for words in longmost_mats:
        render['mostwords'] += "\n" + ','.join([str(i) for i in words])

    ### CHAT TIMING -- only include totals, otherwise hard to see
    users = ['Everyone']
    render['timing_chat'] = ''
    # Make header first from first round
    for k,v in s0['timing']['chat'].items():
        render['timing_chat'] += "Delay"
        for u,c in v.items():
            if (u in users):
                render['timing_chat'] += ",{}".format(u)
        render['timing_chat'] += "\n"
        break
    # Now fill next lines with data
    for k,v in s0['timing']['chat'].items():
        render['timing_chat'] += "{}".format(k)
        for u,c in v.items():
            if (u in users):
                render['timing_chat'] += ",{}".format(c)
        render['timing_chat'] += "\n"

    ### SOCIAL NETWORK
    render['network'] = ''
    for from_user, v in s0['network']['network'].items():
        for to_user, count in v.items():
            render['network'] += "['{}','{}',{:d}],\n".format(from_user,to_user,count)
    # TODO hack drop last comma
    render['network'] = render['network'][:-2]

    return render

def publish(stats, outdir, template, pubfile):
    """
    Publish results, store to disk
    """

    # Prepare Jinja environment and template
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(
        loader=FileSystemLoader('./'),
        autoescape=False
    )
    # select_autoescape(['html', 'xml'])

    template = env.get_template(template)

    # Prep render output
    srender = {}
    for k0, s0 in stats.items():
        srender[k0] = prep_render(s0)
        
    # Make render, use last topic of maximum data range as title
    render = template.render(
        title=srender[max(srender.keys())]['totals']['lasttopic'],
        srender=srender)
        # activetop10=stats[30]['active']['allday'],
        # stats30daily=stats30daily,
        # stats30alltime=stats30alltime)

    # Store output
    with open(os.path.join(outdir, pubfile), 'w') as fd:
        fd.write(render)


def main():
    parser = argparse.ArgumentParser(description='process whatsapp log files into IRC-like statistics page.')

    parser.add_argument('--timeframes', metavar='days', type=int, nargs='+',
                        default=(31, 365, -1),
                        help='timeframe in days to calculate statistics on. -1 for forever')
    parser.add_argument('--parsedlogfile', type=str, metavar='path',
                        help='optional file to store parsed chatlog as CSV')
    parser.add_argument('--chatstatsfile', type=str, metavar='path',
                        help='optional file to store pickled chat statistics to')
    parser.add_argument('--config', type=str, metavar='path',
                        help='optional file with username mappings. If not provided some functionality is lost.')
    parser.add_argument('--template', type=str, metavar='path',
                        help='template file to use', default="pysg_template_tabbed.html")

    parser.add_argument('chatlog', type=str, metavar='chatlog',
                        help='Whatsapp chatlog file to parse')
    parser.add_argument('--debug', action='store_true',
                        help='show debug output')

    # Pre-process command-line arguments
    args = parser.parse_args()
    loglevel = logging.INFO-(args.debug*10)
    logging.basicConfig(level=logging.INFO-args.debug*10, format='%(asctime)s %(message)s')
    logging.debug(args)

    parsecfg = load_config(args.config)

    parse_log(args.chatlog, useraliases=parsecfg['user_aliases'], parsedlogfile=args.parsedlogfile, timeframes=args.timeframes, chatstatsfile=args.chatstatsfile, template=args.template)

class TestSmileyMethods(unittest.TestCase):
    def test_detect(self):
        smiley_test_str3 = ["bla 8 bla", ":‑):‑):‑)", ":)", ":-]", ":]", ":-3", ":3", ":->", ":>", "8-)", "8)", ":-}", ":}", ":o)", ":c)", ":^)", "=]", "=)", ":‑D", ":D", "8‑D", "8D", "x‑D", "xD", "X‑D", "XD", "=D", "=3", ":'‑)", ":')", ":-))", ":‑P", ":P", "X‑P", "XP", "x‑p", "xp", ":‑p", ":p", ":Þ", ":‑þ", ":þ", ":‑b", ":b", "d:", "=p", ";D",">:P"]
        smiley_test_str2 = [":‑(", ":(", ":‑c", ":c", ":‑<", ":<", ":‑[", ":[", ":-||", ">:[", ":{", ":@", ">:(", ":'‑(", ":'("]

        for s in smiley_test_str3:
            print (s, re_smiley_happy.findall(s))

        for s in smiley_test_str2:
            print (s, re_smiley_sad.findall(s))

        smiley_test_str_emoji = ["😀😀😀", "😁", "😂", "😃", "😄", "😅", "😆", "😇", "😈", "😉", "😊", "😋", "😌", "😍", "😎", "😏", "😐", "😑", "😒", "😓", "😔", "😕", "😖", "😗", "😘", "😙", "😚", "😛", "😜", "😝", "😞", "😟", "😠", "😡", "😢", "😣", "😤", "😥", "😦", "😧", "😨", "😩", "😪", "😫", "😬", "😭", "😮", "😯"]
        for s in smiley_test_str_emoji:
            print (s, re_emoji_smiley_happy.findall(s) + re_emoji_smiley_sad.findall(s))

        re_emoji_smiley_sad.findall("".join(smiley_test_str_emoji))
        re_out = re_smiley_happy.findall("".join(smiley_test_str3))
        smileys = {}
        for r in re_out:
            smileys[r] = smileys.get(r,0) + 1
        print(smileys)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == "__main__":
    main()
    exit()
