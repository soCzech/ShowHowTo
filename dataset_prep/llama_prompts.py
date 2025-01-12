def prompt_to_filter_noninstructional_videos(video_title, video_transcript, max_transcript_length=1500):
    """
    Generate a prompt to determine if a video is instructional based on its title and transcript.
    
    Args:
        video_title: The title of the video
        video_transcript: List of transcript segments with start, end, and text fields
        max_transcript_length: Maximum length of transcript to include in prompt
    
    Returns:
        str: Formatted prompt for video classification
    """
    # Extract and join transcript text
    formatted_transcript = " ".join(item['text'] for item in video_transcript)
    
    # Increase length limit for longer videos
    if video_transcript[-1]['end'] > 400:
        max_transcript_length += 2000
        
    # Truncate transcript if needed
    if len(formatted_transcript) > max_transcript_length:
        formatted_transcript = formatted_transcript[:max_transcript_length] + "..."

    prompt = f"""
Based on the following video title and partial transcript segment, determine if the video is instructional in nature, where "instructional" means it involves actively demonstrating or teaching how to perform a specific task or activity with physical steps (e.g., cooking a recipe, repairing something, crafting, etc.). 
Respond with 'Yes' if the video is actively demonstrating or teaching how to perform a specific task, or 'No' if it is not. Then provide a single sentence explanation.

Examples of instructional videos:
- How to Bake a Chocolate Cake
- Reparing a Leaky Faucet
- Learn to Knit a Scarf

Examples of non-instructional videos:
- Discussing Fashion Trends
- Product Reviews and Opinions
- A Vlog of My Daily Life

Example 1:
Video Title: Red Dead Redemption 2 - Herbert Moon and Strange man Eastereggs In Armadillo [SPOILERS]
Partial Video Transcript Segment:
"oh you're back I feared the worst it's all here waiting for you who's that I don't know it's just a little portrait somebody gave me once I always quite liked it why no reason just seem familiar anyway this area is closed to the public if you want to shop here you better act right move you long streak of piss who do you think you are for God's sake get out you degenerate you blew it get out of my store if you don't leave there will be problems okay okay stay calm oh you'll..."

Is this video actively demonstrating or teaching how to perform a specific task? No
Explanation: The video is not actively demonstrating or teaching how to perform a specific task; it appears to be showcasing or discussing Easter eggs in the game Red Dead Redemption 2.

Example 2:
Video Title: Fantastic VEGAN Cupcakes with Raspberry Frosting
Partial Video Transcript Segment:
"hey there I'm chef Annie and tomorrow is Valentine's Day so we are making some extra special cupcakes for this occasion can you believe that we have not made cupcakes on this channel it's about time so today I'm going to show you how to present these cupcakes so they look impressive and absolutely beautiful so enough copy let's cook it so we're going to start by mixing together our wet ingredients..."

Is this video actively demonstrating or teaching how to perform a specific task? Yes
Explanation: The video actively demonstrates and teaches how to make vegan cupcakes with raspberry frosting, as indicated by the detailed steps and instructions given by the chef.

Example 3:
Video Title: How To: Piston Ring Install
Partial Video Transcript Segment:
"hey it's Matt from how to motorcycle repair comm just got done doing a top end on a YZF 250 or yz250 F and I thought I'd do a quick video on how to install a piston ring the easy way now I've done this in the past too but most people will take the ends here and spread it and put it on but you can potentially damage the ring so an easier way to do that is just to take this right here incident in the groove that you need then you bend one up..."

Is this video actively demonstrating or teaching how to perform a specific task? Yes
Explanation: The video is actively demonstrating or teaching how to install a piston ring, which is a specific task.

Example 4:
Video Title: Best gas weed eater reviews Husqvarna 128DJ with 28cc Cycle Gas Powered String Trimmer
Partial Video Transcript Segment:
"Video Transcript: guys i'm shanley today i'm going to tell you about this straight shaft gas-powered trimmer from husqvarna this trimmer runs on a 28 CC two cycle engine it features 1.1 horsepower and a three-piece crankshaft it also has a smart start system as well as an auto return to stop switch and this trimmer is air purge design for easier starting it has a 17 inch cutting path..."

Is this video actively demonstrating or teaching how to perform a specific task? No
Explanation: This video is reviewing the features of a gas-powered trimmer rather than actively demonstrating or teaching how to use it.

Now, determine if the following video is instructional in nature:

Video Title: {video_title}

Partial Video Transcript Segment:
{formatted_transcript}

Is this video actively demonstrating or teaching how to perform a specific task?
    """
    return prompt


def prompt_to_generate_keysteps(video_title, video_transcript):
    """
    Generate a prompt to extract key steps from an instructional video transcript.
    
    Args:
        video_title: The title of the video
        video_transcript: List of transcript segments with start, end, and text fields
    
    Returns:
        str: Formatted prompt for step extraction
    """

    formatted_transcript = ''
    for item in video_transcript:
        formatted_transcript += f"{item['start']:.2f} - {item['end']:.2f}: \"{item['text']}\"\n"
    
    prompt = f"""
Below are transcripts from YouTube instructional videos and their corresponding extracted steps in a clear, third-person, step-by-step format like WikiHow. Each step is concise, actionable, and temporally ordered as they occur in the video. The steps include start and end timestamps indicating when the steps are carried out in the video. Follow this format to extract and summarize the key steps from the provided transcript.

Example 1:
YouTube Video Title:
"BÁNH TÁO MINI - How To Make Apple Turnovers  | Episode 11 | Taste From Home"

YouTube Video Transcript:
[start of transcript]
[start of transcript]
0.87 - 7.79: "Hey little muffins, today we will make together a super easy, quick and delicious apple turnovers."
7.79 - 9.35: "40 minutes for all the process."
9.35 - 11.95: "Seriously, can someone deny them?"
11.95 - 13.63: "Ok, let's begin."
13.63 - 18.82: "First of all, combine the apple cubes, lemon juice, cinnamon and sugar in a bowl."
26.69 - 29.59: " Mixing, mixing, mixing."
29.59 - 32.62: "Apple and cinnamon always go perfectly together."
32.62 - 43.52: "Now using a round cutter or glass like me, cut 15 rounds from the pastry sheet."
57.86 - 64.99: " Here comes the fun part."
64.99 - 69.97: "Spoon about 2 teaspoons apple mixture in the center of one round."
69.97 - 74.41: "Using your fingers, gently fold the pastry over to enclose filling."
88.47 - 104.48: " After that, use a fork and press around the edges to seal and make your apple turnovers look more beautiful."
104.48 - 105.84: "This is how it looks like."
109.99 - 113.53: " I will show you one more time to make sure that you understand the technique."
113.53 - 117.20: "And if you still find my apple turnovers too ugly, I'm really sorry."
117.20 - 121.66: "Anyways, just have fun making them with your family and friends."
121.66 - 124.43: "Then it doesn't matter if they are beautiful or not."
124.43 - 125.45: "At least they are tasty."
127.35 - 133.35: " I finally finished my 15 turnovers."
151.62 - 157.46: " Now we will lightly beat one egg in a small bowl and then egg wash our apple turnovers."
157.46 - 164.10: "The egg wash will give them a gorgeous light brown color after baking in the oven."
164.10 - 174.87: "We will bake the apple turnovers at 180°C in about 18-20 minutes until golden."
174.87 - 177.35: "Your kitchen should smell amazing by now."
178.17 - 181.33: " Okay, taking photos like this is like a torture."
181.33 - 185.55: "I just can't resist grabbing one of them and enjoying immediately."
185.55 - 186.17: "Yay!"
186.17 - 189.00: "So finally I can try my apple turnovers."
189.00 - 201.34: "I'm really really excited."
201.34 - 203.22: "Oh my god!"
203.22 - 204.46: "This is really really good."
204.46 - 207.26: "You guys should definitely try out this recipe."
208.69 - 215.49: " So I hope you enjoyed this video and I will see you in my next video for a new recipe."
215.49 - 216.03: "Bye guys!"
[end of transcript]

Extracted Steps:
[
  {{ "WikiHow Title": "How to Make Apple Turnovers" }},
  {{ "steps": [
    {{ "step": 1, "instruction": "Combine apple cubes, lemon juice, cinnamon, and sugar in a bowl.", "start_timestamp": 13.63, "end_timestamp": 18.82 }},
    {{ "step": 2, "instruction": "Mix the ingredients thoroughly.", "start_timestamp": 26.69, "end_timestamp": 29.59 }},
    {{ "step": 3, "instruction": "Cut 15 rounds from the pastry sheet using a round cutter or a glass.", "start_timestamp": 32.62, "end_timestamp": 43.52 }},
    {{ "step": 4, "instruction": "Spoon about 2 teaspoons of the apple mixture into the center of one round.", "start_timestamp": 64.99, "end_timestamp": 69.97 }},
    {{ "step": 5, "instruction": "Gently fold the pastry over to enclose the filling using your fingers.", "start_timestamp": 69.97, "end_timestamp": 74.41 }},
    {{ "step": 6, "instruction": "Press around the edges with a fork to seal and beautify the turnovers.", "start_timestamp": 88.47, "end_timestamp": 104.48 }},
    {{ "step": 7, "instruction": "Repeat the technique until all turnovers are formed.", "start_timestamp": 109.99, "end_timestamp": 113.53 }},
    {{ "step": 8, "instruction": "Lightly beat one egg in a small bowl.", "start_timestamp": 151.62, "end_timestamp": 157.46 }},
    {{ "step": 9, "instruction": "Egg wash the apple turnovers to give them a gorgeous light brown color after baking.", "start_timestamp": 157.46, "end_timestamp": 164.10 }},
    {{ "step": 10, "instruction": "Bake the apple turnovers at 180°C for 18-20 minutes until golden.", "start_timestamp": 164.10, "end_timestamp": 174.87 }},
    {{ "step": 11, "instruction": "Enjoy the freshly baked apple turnovers.", "start_timestamp": 178.17, "end_timestamp": 185.55 }}
    ]
  }}
]

Example 2:
YouTube Video Title:
"How to Clean Your Car Tires & Whitewalls"

YouTube Video Transcript:
[start of transcript]
0.17 - 5.42: " Cleaning your tires properly requires more than just using car wash soap and then rinsing it down."
5.42 - 11.84: "Today I want to show you the proper way to clean your tires to give it that nice new natural darkened look."
17.69 - 22.31: " Now, although you may be sitting there thinking to yourself, do I really need a specific tire cleaner?"
22.31 - 25.17: "Well, if you're an auto detailer, we highly recommend it."
25.17 - 30.74: "But even if you're a car enthusiast, there's a few reasons why this type of product can come in handy."
30.74 - 39.78: "Now, our tire and white wall cleaner is super concentrated, making it aggressive enough to really break down any of that gunk that gets stuck to the rubber of your tire."
39.78 - 45.30: "Not to mention, can safely and really help you clean any of those white walls or white letters."
45.80 - 55.29: " One ingredient that really makes this product unique is that it actually contains darkening agents that really help to bring out that dark, rich color of the rubber."
55.29 - 59.41: "This can also help with assisting when doing your tire dressing."
59.41 - 65.24: "It allows the tire dressing to set up properly, giving you a nice shine, and helps it to last even longer."
65.62 - 75.37: " Now, the first thing we want to do here is not just rinse down the tire, but as well as some of the surrounding areas like the wheel, wheel well, as well as the body of the vehicle here."
95.34 - 103.47: " Now that we've thoroughly rinsed down the surrounding areas as well as the tire, we can go ahead and take our tire and whitewall and spray an even mist all on the tire."
109.80 - 113.99: " Now, what we want to do is we want to allow the product to set up for about 30 seconds."
113.99 - 120.05: "This is really going to help to break down any of that gunk or dirt that is just stuck onto the rubber of the tire here."
120.05 - 125.30: "I mean, as you can already see, just by applying it on within a few seconds here, it's already turning brown."
125.30 - 129.98: "So it's showing that the product is activating, pulling off any of that dirt."
129.98 - 132.70: "So now we've allowed the product to set up for about 30 seconds or so."
132.70 - 137.59: "We can go ahead, take our brush, dip it in a bucket of water, and then begin with the scrubbing process."
156.52 - 163.69: " Seeing all these brown suds just like this is exactly what you want to see when using this product."
163.69 - 173.68: "Again that's showing that it's activated and it's really breaking down everything that is just on the rubber of this tire."
173.68 - 176.08: "Now we've scrubbed it we can go ahead and rinse it right down."
196.36 - 208.82: " So as you guys can see, it's honestly just as simple and as easy as that, with just a little bit of product, a little bit of scrubbing, and then a simple rinse, we were able to really bring back that dark, rich color to the rubber of this tire once again."
208.82 - 222.94: "So if this video was helpful, make sure you definitely give it a thumbs up, subscribe to our channel for more product and how-to videos, and to order your very own Tire and White Wall Cleaner, make sure you visit us at detailking.com, where we have everything you need to keep your car clean like a detail king."
222.94 - 223.84: "We'll see you guys next time."
[end of transcript]

Extracted Steps:
[
  {{ "WikiHow Title": "How to Clean Your Car Tires & Whitewalls"}},
  {{ "steps": [
    {{ "step": 1, "instruction": "Rinse down the tire and surrounding areas such as the wheel, wheel well, and the body of the vehicle.", "start_timestamp": 65.62, "end_timestamp": 75.37 }},
    {{ "step": 2, "instruction": "Spray an even mist of tire and whitewall cleaner all over the tire.", "start_timestamp": 95.34, "end_timestamp": 103.47 }},
    {{ "step": 3, "instruction": "Allow the cleaner to set up for about 30 seconds to break down dirt and gunk on the tire.", "start_timestamp": 109.80, "end_timestamp": 120.05 }},
    {{ "step": 4, "instruction": "Observe the cleaner turning brown, indicating it is working.", "start_timestamp": 120.05, "end_timestamp": 125.30 }},
    {{ "step": 6, "instruction": "Dip the brush in a bucket of water and begin scrubbing the tire.", "start_timestamp": 132.70, "end_timestamp": 137.59 }},
    {{ "step": 7, "instruction": "Check for brown suds while scrubbing, which shows the cleaner is breaking down dirt.", "start_timestamp": 156.52, "end_timestamp": 163.69 }},
    {{ "step": 8, "instruction": "Rinse the tire thoroughly to remove all cleaner and dirt.", "start_timestamp": 173.68, "end_timestamp": 176.08 }},
    {{ "step": 9, "instruction": "Enjoy the clean, dark, and rich color of your tire's rubber.", "start_timestamp": 196.36, "end_timestamp": 208.82 }}
   ]
  }}
]

Now, extract the steps from the following transcript:

YouTube Video Title:
{video_title}

YouTube Video Transcript:
[start of transcript]
{formatted_transcript}
[end of transcript]

Extracted Steps:
```json
"""
    return prompt


if __name__ == "__main__":
    # An example video and its transcript
    example_title = "How to Baked a Potato in the pressure cooker."
    example_transcript = [
        {
            "start": 0.504,
            "end": 2.605,
            "text": " Hi, it's Matthew in his pressure cooker again."
        },
        {
            "start": 2.605,
            "end": 6.086,
            "text": "And today I'm going to make baked potatoes in the pressure cooker."
        },
        {
            "start": 6.086,
            "end": 11.708,
            "text": "Obviously they're not baked potatoes, just an easy way to make something like a baked potato."
        },
        {
            "start": 11.708,
            "end": 13.289,
            "text": "It's basically going to be steaming them."
        },
        {
            "start": 13.289,
            "end": 18.051,
            "text": "So I use my aluminum foil trick."
        },
        {
            "start": 18.051,
            "end": 19.532,
            "text": "Put some aluminum foil in there."
        },
        {
            "start": 19.532,
            "end": 27.835,
            "text": "Now a lot of pressure cookers do come with standoffs so that you can put things in where they're sitting above the water."
        },
        {
            "start": 28.857,
            "end": 29.698,
            "text": " I'm going to make two little rings."
        },
        {
            "start": 29.698,
            "end": 38.503,
            "text": "I'm going to take my potatoes, put them in so that they're going to be steamed nicely."
        },
        {
            "start": 38.503,
            "end": 40.424,
            "text": "Because you don't want to actually boil them."
        },
        {
            "start": 40.424,
            "end": 42.285,
            "text": "You want to steam them."
        },
        {
            "start": 42.285,
            "end": 44.987,
            "text": "And I've got my cup and a half of water."
        },
        {
            "start": 44.987,
            "end": 50.23,
            "text": "I'll just dump that in."
        },
        {
            "start": 50.23,
            "end": 52.272,
            "text": "See what I've got in here?"
        },
        {
            "start": 52.272,
            "end": 53.472,
            "text": "So they're just sitting there."
        },
        {
            "start": 53.472,
            "end": 54.313,
            "text": "They're above the water."
        },
        {
            "start": 54.313,
            "end": 57.495,
            "text": "Now, the amount of time varies quite a bit."
        },
        {
            "start": 59.853,
            "end": 62.054,
            "text": " For these ones I'm going to put them in for about 15 minutes."
        },
        {
            "start": 62.054,
            "end": 63.755,
            "text": "Smaller potatoes maybe a little bit less."
        },
        {
            "start": 63.755,
            "end": 65.016,
            "text": "Larger potatoes a little bit more."
        },
        {
            "start": 65.016,
            "end": 68.518,
            "text": "The nice thing about this is it doesn't tie up anything else."
        },
        {
            "start": 68.518,
            "end": 69.898,
            "text": "It doesn't heat up your kitchen."
        },
        {
            "start": 69.898,
            "end": 73.881,
            "text": "It's really good for in the summer when you want to have something like a baked potato."
        },
        {
            "start": 73.881,
            "end": 75.481,
            "text": "It's just going to be steamed."
        },
        {
            "start": 75.481,
            "end": 76.082,
            "text": "So there it is."
        },
        {
            "start": 76.082,
            "end": 76.942,
            "text": "15 minutes."
        },
        {
            "start": 76.942,
            "end": 80.484,
            "text": "When they're done you just pop them out and eat them like a normal baked potato."
        },
        {
            "start": 80.484,
            "end": 81.465,
            "text": "I hope you find this useful."
        },
        {
            "start": 81.465,
            "end": 87.928,
            "text": "If you want to hear more ideas or have any questions leave a comment, send me an email and I'll see what I can do for you."
        },
        {
            "start": 89.452,
            "end": 92.674,
            "text": " I hope you're enjoying your pressure cooker as much as I'm enjoying mine."
        },
        {
            "start": 92.674,
            "end": 93.915,
            "text": "Bye."
        },
        {
            "start": 93.915,
            "end": 96.837,
            "text": "And here we go."
        },
        {
            "start": 96.837,
            "end": 101.699,
            "text": "I'll pull it out."
        },
        {
            "start": 101.699,
            "end": 104.601,
            "text": "There is the baked potato or steamed potato actually."
        },
        {
            "start": 104.601,
            "end": 107.523,
            "text": "It's nice and it's pretty dry on the outside."
        },
        {
            "start": 108.55,
            "end": 110.371,
            "text": " Just cut it open."
        },
        {
            "start": 110.371,
            "end": 112.431,
            "text": "It's nice and soft."
        },
        {
            "start": 112.431,
            "end": 115.072,
            "text": "Well cooked on the inside."
        },
        {
            "start": 115.072,
            "end": 117.412,
            "text": "Great to cook it up however you're going to make a meal."
        },
        {
            "start": 117.412,
            "end": 119.353,
            "text": "If you're going to eat it like a traditional baked potato."
        },
        {
            "start": 119.353,
            "end": 121.633,
            "text": "If you're going to use it for potato salad or whatever."
        },
        {
            "start": 121.633,
            "end": 123.794,
            "text": "I hope you enjoyed it."
        },
        {
            "start": 123.794,
            "end": 126.755,
            "text": "If you want to see any other ideas, check my channel."
        },
        {
            "start": 126.755,
            "end": 128.355,
            "text": "See what other things I've got posted."
        },
        {
            "start": 128.355,
            "end": 134.157,
            "text": "If you've got ideas that you don't know how to do, send me an email or leave a comment and I'll see what I can do."
        },
        {
            "start": 134.157,
            "end": 134.697,
            "text": "Hope you enjoyed it."
        },
        {
            "start": 134.697,
            "end": 135.037,
            "text": "Bye."
        }
    ]

    print("=== Llama 3 prompt to filter non-instructional videos ===")
    print(prompt_to_filter_noninstructional_videos(example_title, example_transcript))

    print("\n" + "=" * 80 + "\n")

    print("=== Llama 3 prompt to generate keysteps ===")
    print(prompt_to_generate_keysteps(example_title, example_transcript))
