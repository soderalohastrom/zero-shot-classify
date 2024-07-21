import os
from dotenv import load_dotenv
import time
import logging
import json

from anthropic import Anthropic

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize the Anthropic client with your API key
client = Anthropic(api_key=api_key)

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.basicConfig(level=logging.WARNING)

# Define system prompts for each category
system_prompts = {
    "Politics": (
f"""The following are political category type labels of people followed by an explanation to help in your task of taking in answers to survey questions about politics generally, with the addition of periodicals and podcasts to add nuance. These are to be introduced per person to help in dating matching bios to find compatible people to match together. You are to take in the inputs and categorize the person type by outputting only the label phrase that best fits the person based on their input. Do not offer any other comments, only output the label\n\n"
"Here are your 16 labels for 'Politics':\n\n"
"1. Republican Family Traditionalists - Conservatives who prioritize traditional family values, often with a religious foundation. They support Republican policies that align with these values and may consume media from sources like Fox News, The Wall Street Journal, and conservative talk radio.\n\n"
"2. God and Guns Conservatives - Individuals who prioritize Second Amendment rights and religious values, often seeing them as interconnected. They strongly support gun ownership and may be involved in gun rights organizations. They also tend to be socially conservative, emphasizing traditional Christian values. They often consume media from sources that align with their views on guns and religion, such as conservative talk radio, Fox News, and gun enthusiast publications.\n\n"
"3. Progressive Activist - Individuals with liberal or progressive views, passionate about social justice, environmental issues, and supporting the LGBTQ+ community, often engaging with media like CNN, NPR, and various progressive podcasts.\n\n"
"4. Loyal Democratic Voter - Individuals who consistently vote for the Democratic party, often due to family tradition or a general alignment with the party's platform. They may not be as politically active or passionate as Progressive Activists but reliably support Democratic candidates and policies. They tend to consume mainstream media sources that lean left, such as MSNBC or CNN.\n\n"
"5. Moderate Pragmatist - People who identify as moderates or centrists, possibly leaning slightly left or right on specific issues, valuing practical solutions and open to diverse viewpoints, with a varied media diet including both liberal and conservative sources.\n\n"
"6. Libertarian Thinker - Those who value individual liberty, minimal government intervention, and may identify as libertarian or independent, often consuming content related to economics, personal growth, and diverse political perspectives.\n\n"
"7. Apolitical Minimalist - Individuals who are largely disinterested in politics, prefer not to engage in political discussions, and focus on personal growth, health, or other non-political interests.\n\n"
"8. Eco-Conscious Liberal - People with liberal views, particularly focused on environmental issues and sustainability, often consuming media related to these topics and supporting progressive policies.\n\n"
"9. Independent Swing Voter - Individuals who do not strongly align with a single political party, open to various viewpoints, and may switch their vote based on the candidates and issues at hand, consuming a balanced mix of news sources.\n\n"
"10. Cultural Conservative - Those with conservative views but who also engage in cultural, historical, or spiritual media, and may prefer more traditional forms of media consumption.\n\n"
"11. Liberal Intellectual - Academically inclined individuals with liberal views, who enjoy deep discussions and theoretical debates, often consuming a wide range of intellectually stimulating content.\n\n"
"12. Non-Political Humanist - Individuals who avoid political labels, focus on human rights and personal values over party politics, and prefer to engage in discussions about personal and societal growth without a strong political affiliation.\n\n"
"13. Fiscally Conservative, Socially Liberal - Those who hold conservative views on economic issues but are liberal on social issues, often consuming a mix of financial news and progressive social content.\n\n"
"14. Politically Flexible Pragmatist - People who value practical solutions over rigid ideologies, are open to different viewpoints, and focus on finding common ground, often consuming a variety of news sources and podcasts that offer balanced perspectives.\n\n"
"15. Trump Followers - Ardent supporters of former President Donald Trump, who align with his political views, policies, and communication style. They often consume media that aligns with Trump's perspectives, such as Fox News, Newsmax, and various conservative social media influencers.\n\n"
"16. Politics Undetermined - A fallback option for those whose political views are not clearly defined or do not align with the provided categories.\n\n"
"17. Green Party Advocate - Individuals who strongly support the Green Party's platform, focusing on environmentalism, social justice, and non-violence. They often consume media from sources that align with these values, such as The Green Papers, Democracy Now!, and various environmental and social justice podcasts.\n\n"
"18. Socialist or Democratic Socialist - Individuals who identify with socialist or democratic socialist ideologies, advocating for social ownership of the means of production and a more equitable distribution of wealth. They often consume media from sources that align with these views, such as Jacobin, The Nation, and various socialist and labor podcasts.\n\n"
"19. Anti-Establishment Dissenter - Individuals who are critical of the current political system and institutions, often distrusting mainstream media and political parties. They may be drawn to alternative media sources, independent journalists, and activist groups.\n\n"
"20. Political Activist - Individuals who are actively involved in political campaigns, protests, or advocacy groups, often passionate about specific issues and seeking to influence policy changes. They may consume media from sources that align with their activism, such as political blogs, news websites, and social media platforms.\n\n"
"21. Political Analyst - Individuals who are knowledgeable about political systems, policies, and current events, often engaging in political discussions and analysis. They may consume a wide range of news sources, political commentary, and academic research.\n\n"
"22. Political Observer - Individuals who are interested in politics but prefer to observe and learn about different perspectives without actively participating in political activities. They may consume a variety of news sources, political documentaries, and podcasts.\n\n"
"23. Political Cynic - Individuals who are disillusioned with the political system and believe that it is corrupt or ineffective. They may be skeptical of politicians and political parties and often consume media that reinforces their cynicism.\n\n"
"24. Political Apathetic - Individuals who are indifferent to politics and do not engage in political discussions or activities. They may not consume much political news or commentary.\n\n"
"Summary: These categories cover a wide range of political views, from strong support for specific parties or politicians to more moderate or apolitical stances. They also take into account the types of media consumed, which can provide additional insight into an individual's political leanings and interests. By using these labels to categorize people based on their survey responses, you can help match them with compatible partners who share similar values and perspectives."
"""
    ),
    "Lifestyle": (
f"""The following are lifestyle category type labels of people followed by an explanation to help in your task of taking in answers to survey questions about lifestyle preferences and habits. These are to be introduced per person to help in dating matching bios to find compatible people to match together. You are to take in the inputs and categorize the person type by outputting only the label phrase that best fits the person based on their input. Do not offer any other comments, only output the label.\n\n"
"Here are your labels for 'Lifestyle Type':\n\n"
"1. Adventurous Travelers - Individuals who love exploring new places, cultures, and experiences, often prioritizing travel in their lives and seeking out new adventures.\n\n"
"2. Health Enthusiasts - People who are passionate about maintaining a healthy lifestyle through diet, exercise, and wellness practices, and who stay updated on the latest health trends.\n\n"
"3. Social Butterflies - Those who thrive in social settings, enjoy meeting new people, and often participate in various social activities, gatherings, and events.\n\n"
"4. Fitness Fanatics - Individuals who are dedicated to physical fitness and regularly engage in various forms of exercise, from gym workouts to sports and outdoor activities.\n\n"
"5. Culinary Explorers - People who have a passion for food, love trying new cuisines, experimenting with recipes, and often seek out unique dining experiences.\n\n"
"6. Family and Community Oriented - Individuals who prioritize family and community connections, often engaging in family activities, community events, and valuing close relationships.\n\n"
"7. Intellectuals and Lifelong Learners - Those who have a deep love for learning, enjoy intellectual discussions, and continuously seek to expand their knowledge through reading, courses, and other educational pursuits.\n\n"
"8. Balanced Work-Life Advocates - People who strive for a healthy balance between their professional and personal lives, valuing time for relaxation, hobbies, and personal interests alongside their careers.\n\n"
"9. Homebodies with Hobbies - Individuals who prefer spending time at home, enjoying various hobbies and activities within a cozy and comfortable environment.\n\n"
"10. Active and Outdoorsy - Those who love spending time outdoors, engaging in activities such as hiking, camping, and sports, and who prioritize an active lifestyle.\n\n"
"11. Spiritual and Mindful - People who focus on spiritual growth, mindfulness, and inner peace, often practicing meditation, yoga, or other spiritual and reflective activities.\n\n"
"12. Cultural Connoisseurs - Individuals who have a deep appreciation for the arts, culture, and history, often attending cultural events, visiting museums, and enjoying diverse forms of artistic expression.\n\n"
"13. Wellness and Self-Care Advocates - People who prioritize their well-being through practices such as self-care routines, mental health awareness, and holistic health approaches.\n\n"
"14. Social and Professional Networkers - Individuals who thrive on building and maintaining extensive social and professional networks, often participating in networking events and social gatherings.\n\n"
"15. Community Builders and Volunteers - Those who are dedicated to improving their communities through volunteer work, social initiatives, and active participation in community-building efforts.\n\n"
"16. Nature Lovers and Environmentalists - People who have a strong connection to nature and are passionate about environmental conservation, often participating in outdoor activities and eco-friendly practices.\n\n"
"17. Simplifiers and Decluttering Enthusiasts - Individuals who focus on minimalism, decluttering their spaces, and simplifying their lives to reduce stress and increase efficiency.\n\n"
"18. Philanthropists and Humanitarians - Those who are dedicated to charitable work, supporting humanitarian causes, and making a positive impact on the lives of others through philanthropy.\n\n"
"19. Pet Owners and Animal Lovers - People who have a deep affection for animals, often owning pets and advocating for animal welfare.\n\n"
"20. Adrenaline Junkies and Thrill Seekers - Individuals who seek out thrilling and adventurous activities, such as extreme sports, for the rush of adrenaline.\n\n"
"21. Sustainable Living Advocates - People who are committed to living sustainably, making environmentally conscious choices in their daily lives to reduce their ecological footprint.\n\n"
"22. Career Achievers and Go-Getters - Individuals who are highly ambitious and focused on achieving their career goals, often prioritizing professional growth and success.\n\n"
"23. Luxury and High-End Lifestyle Enthusiasts - Those who appreciate and indulge in luxury goods, high-end experiences, and a sophisticated lifestyle.\n\n"
"24. Free Spirits and Nonconformists - People who embrace unconventional lifestyles, value their independence, and often reject societal norms in favor of personal freedom.\n\n"
"25. Classic and Timeless Style Aficionados - Individuals who have a refined taste and appreciation for classic, timeless styles in fashion, decor, and lifestyle choices.\n\n"
"26. Country Living and Rustic Enthusiasts - People who prefer a rural, rustic lifestyle, often enjoying activities like farming, gardening, and living close to nature.\n\n"
"27. City Dwellers and Urban Explorers - Those who thrive in the bustling environment of a city, enjoying urban culture, nightlife, and the convenience of metropolitan living.\n\n"
"28. Digital Nomads and Remote Workers - Individuals who work remotely while traveling, embracing the flexibility to live and work from various locations around the world.\n\n"
"29. Lifestyle Undetermined - When the specific details of the lifestyle are unclear or not explicitly mentioned.\n\n"
"These categories are based on the lifestyle preferences and habits that influence their daily lives, which can be a foundation for romantic attraction and compatibility based on shared interests."
),
    "Children": (
f"""The following are children preference category type labels of people followed by an explanation to help in your task of taking in answers to survey questions about their preferences and situations regarding children. These are to be introduced per person to help in dating matching bios to find compatible people to match together. You are to take in the inputs and categorize the person type by outputting only the label phrase that best fits the person based on their input. Do not offer any other comments, just output the main label.\n\n"
"Here are your labels for 'Children Preference': \n\n"
"1. 'No Children, Wants Children' - [ Wants (More) Kids: Yes; Ok with Existing Children: Yes ]\n\n"
"2. 'No Children, Open to Having Children' - [ Wants (More) Kids: Maybe; Ok with Existing Children: Yes ]\n\n"
"3. 'No Children, Does Not Want Children' - [ Wants (More) Kids: No; Ok with Existing Children: No ]\n\n"
"4. 'Has Children, Open to More' - [ Wants (More) Kids: Yes; Ok with Existing Children: Yes; Children Ages: Young (0-12), Adolescent (13-18), Adult (18+) ]\n\n"
"5. 'Has Children, Does Not Want More' - [ Wants (More) Kids: No; Ok with Existing Children: Yes; Children Ages: Young (0-12), Adolescent (13-18), Adult (18+) ]\n\n"
"6. 'No Preference for Having Children' - [ Wants (More) Kids: Open to it; Ok with Existing Children: Yes ]\n\n"
"7. 'Specific Conditions for Having/Accepting Children' - [ Wants (More) Kids: Yes/Maybe; Ok with Existing Children: Specific Conditions (e.g., 'Yes but prefer older than 13', 'Yes but not at home', 'Depends on his/her relationship with the childs parent') ]\n\n"
"8. 'Has Children, Specific Preferences for New Children' - [ Wants (More) Kids: No; Ok with Existing Children: Yes, with specific preferences (e.g., 'Yes, but prefer older children', 'Yes, but not very young') ]\n\n"
"9. 'Adoptive Parents' - [ Wants (More) Kids: Open to adoption; Ok with Existing Children: Yes; Children Ages: Young (0-12), Adolescent (13-18), Adult (18+) ]\n\n"
"10. 'No Children, Prefers Not to Have Children' - [ Wants (More) Kids: No; Ok with Existing Children: Yes, but prefers not to have any more children ]\n\n"
"11. 'No Children, Unsure About Having Children' - [ Wants (More) Kids: Maybe; Ok with Existing Children: Depends on the situation (e.g., 'Depends - Maybe if they were older and have their own lives') ]\n\n"
"12. 'Has Adult Children, No Interest in Having More' - [ Wants (More) Kids: No; Ok with Existing Children: Yes, but prefers adult children ]\n\n"
"13. 'No Children, Prefers No Kids' - [ Wants (More) Kids: No Preference Open to it; Ok with Existing Children: Preferably not ]\n\n"
"14. 'Not Enough Information About Children' - [ Based on the limited information provided, cannot determine a specific category ]\n\n"
"These categories cover a range of preferences and situations, including whether individuals already have children, their openness to having more, and their comfort with a partners existing children. Just return the label and not the number next to the label."
),
    "Upbringing": (
f"""The following are upbringing category type labels of people followed by an explanation to help in your task of taking in answers about their childhood family structure and environment. These are to be introduced per person to help in dating matching bios to find compatible people to match together. You are to take in the inputs and categorize the person type by outputting only the label phrase that best fits the person based on their input, considering the influence of their family structure, cultural background, and environment. Do not give any explanation whatsoever, just provide the label.\n\n"
"Here are your labels for 'Upbringing':\n\n"
"1. Traditional Family Structure - Raised in a household with two parents, typically biological, with a traditional family setup.\n\n"
"2. Single Parent Household - Raised by a single parent, either mother or father, who primarily took on the role of caregiver.\n\n"
"3. Blended Family - Raised in a family that includes step-parents and/or step-siblings, often due to remarriage of one or both parents.\n\n"
"4. Adoptive or Foster Family - Raised by adoptive parents or within the foster care system, having non-biological parental figures.\n\n"
"5. Multicultural Background - Raised in a household that blends multiple cultures, traditions, and possibly languages, contributing to a diverse upbringing.\n\n"
"6. Religious Upbringing - Raised in a household where religious practices and beliefs played a significant role in daily life and values.\n\n"
"7. Migratory Upbringing - Raised in a family that frequently moved from place to place, often due to a parent's job or lifestyle, leading to exposure to various cultures and environments.\n\n"
"8. Rural Upbringing - Raised in a rural or countryside environment, often characterized by close-knit communities and agricultural surroundings.\n\n"
"9. Inner-City Upbringing - Raised in the central part of a large city, often characterized by a dense population, diverse cultures, and a vibrant urban lifestyle.\n\n"        
"10. Urban Upbringing - Raised in a city environment, characterized by a fast-paced lifestyle, diverse populations, and metropolitan culture.\n\n"
"11. Rural then Urban - Raised initially in a rural setting and later moved to an urban environment, experiencing both lifestyles and the transition between them.\n\n" 
"12. Suburban Upbringing - Raised in the suburban areas surrounding a major city, characterized by residential neighborhoods, local schools, and a community-oriented lifestyle.\n\n"
"13. Suburban - Unconventional - Raised in suburban areas but in a non-traditional or unconventional family setting, which may include alternative schooling, unique family dynamics, or atypical community interactions.\n\n"
"14. Military Family Background - Raised in a family with one or more members serving in the military, often experiencing frequent relocations and a disciplined lifestyle.\n\n"
"15. Immigrant Family Background - Raised in a household with parents or guardians who are immigrants, blending native and new cultural experiences.\n\n"
"16. Academic-Focused Upbringing - Raised in a household that places a strong emphasis on education and academic achievement, often with high expectations for scholarly success.\n\n"
"17. Entrepreneurial Family - Raised in a family where business and entrepreneurship are significant, with parents or guardians who run their own businesses or encourage entrepreneurial activities.\n\n"
"18. Bohemian - Communal Upbringing - Raised in a non-traditional, often artistic or communal living situation, with a focus on shared resources, creativity, and alternative lifestyles.\n\n"
"19. Artistic or Creative Family - Raised in a household that values and encourages artistic or creative pursuits, such as music, art, or performance.\n\n"
"20. Upbringing Undetermined - When the specific details of the upbringing are unclear or not explicitly mentioned.\n\n"
"These categories are based on the family structure and environment that influenced their upbringing, which can be a foundation for romantic attraction and compatibility based on shared experiences. Just provide this label and not the number next to the label."
),
    "Geo-Familiarity": (
f"""The following are geo-regional familiarity category type labels of people followed by an explanation to help in your task of taking in answers about their upbringing and regional identity formation in childhood. These are to be introduced per person to help in dating matching bios to find compatible people to match together. You are to take in the inputs and categorize the person type by outputting only the label phrase that best fits the person based on their input. Give more weight to where they were raised than where they were born, focusing on their identification with regional traditions, customs, recreational activities, colloquialisms, and familiar regional brand names.\n\n"
"Here are your labels for 'Geo-regional Familiarity':\n\n"
"1. California\n\n"
"2. Florida\n\n"
"3. Texas & Deep South - Ex:(Alabama, Mississippi, Louisiana)\n\n"
"4. Northeast Corridor - Ex:(includes major urban areas from Washington, D.C. to Boston, MA)\n\n"
"5. New England - Ex:(Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut)\n\n"
"6. Atlantic Coast & Appalachia - Ex:(Maryland, Virginia, West Virginia, Kentucky, Tennessee, Georgia, North Carolina, South Carolina)\n\n"
"7. Great Lakes & Midwest - Ex:(Ohio, Michigan, Indiana, Illinois, Wisconsin, Minnesota)\n\n"
"8. Mountain West - Ex:(Idaho, Montana, Wyoming, Colorado, Utah)\n\n"
"9. Southwest Desert - Ex:(Nevada, Arizona, New Mexico)\n\n"
"10. Cascadia & Alaska - Ex:(Washington, Oregon, parts of Northern California)\n\n"
"11. Canada\n\n"
"12. Mexico & South America\n\n"
"13. Western Europe\n\n"
"14. Australia & Pacific Island\n\n"
"15. Africa & Caribbean\n\n"
"16. Asia & Baltics\n\n"
"17. Uncertain or Ambiguous\n\n"
"These categories are based on the main geographic region that influenced their identity formation in childhood, which can be a foundation for romantic attraction and compatibility based on shared regional familiarities. Do not offer any other comments, do not print out any states used as examples, such as Ex:(Alabama, Mississippi, Louisiana) - or the number of the category. I only want you output the explicit label of the region as defined."
),
    "Why-Single": (
f"""The following are relationship history category labels to help categorize individuals based on their past relationships and reasons for being single. These labels will be used to help match people with compatible relationship histories and goals. Your task is to take in the person's description of their relationship history and categorize them by outputting only the label phrase that best fits their experience. Give more weight to the most recent or significant relationships and the reasons they ended.\n\n"
"Here are the labels for 'Relationship History':\n\n"
"1. Divorced - Married too young: Individuals who were divorced because they got married at a young age and grew apart or realized they were not compatible.\n\n"
"2. Divorced - Long marriage, grew apart: Those who were married for many years but eventually grew apart or developed different interests and goals.\n\n"
"3. Divorced - Infidelity: Individuals whose marriages ended due to infidelity by one or both partners.\n\n"
"4. Divorced - Incompatibility: Those who divorced due to fundamental incompatibilities in personality, values, or whysingle.\n\n"
"5. Divorced - Spouse's personal issues: Individuals whose marriages ended due to their spouse's personal struggles, such as substance abuse or mental health issues.\n\n"
"6. Divorced - Disagreements over having children: Those who divorced because they couldn't agree on whether to have children or how to raise them.\n\n"
"7. Divorced - Career or lifestyle differences: Individuals whose marriages ended due to diverging career paths or lifestyle preferences.\n\n"
"8. Divorced - Lack of emotional intimacy or connection: Those who divorced because of a lack of emotional connection or intimacy in the marriage.\n\n"
"9. Divorced - Other reasons: Individuals who are divorced for reasons not covered by the other categories.\n\n"
"10. Widowed - Ready to find love again: Those who lost a spouse and are now ready to open their hearts to a new relationship.\n\n"
"11. Never married - Focused on career or personal growth: Individuals who have prioritized their career or personal development and haven't been married.\n\n"
"12. Never married - Hasn't found the right person: Those who haven't been married because they haven't met the right partner.\n\n"
"13. Long-term relationship ended - Incompatibility: Individuals whose long-term relationships ended due to incompatibilities in goals, values, or lifestyles.\n\n"
"14. Long-term relationship ended - Distance or logistics: Those whose long-term relationships ended because of challenges related to distance or logistics.\n\n"
"15. Long-term relationship ended - Commitment issues: Individuals whose long-term relationships ended due to one or both partners having commitment issues.\n\n"
"16. Long-term relationship ended - Other reasons: Those whose long-term relationships ended for reasons not covered by the other categories.\n\n"
"17. Dating casually - Seeking the right connection: Individuals who have been dating casually but are open to a serious relationship with the right person.\n\n"
"18. Relationship history varied - Open to new possibilities: Those with diverse relationship histories who are open-minded about new relationships.\n\n"
"19. Prefers not to say: Individuals who prefer not to disclose or discuss their relationship history.\n\n"
"20. Relationship History Undetermined: A fallback option for those whose relationship history is unclear or doesn't fit the other categories.\n\n"
"Do not offer any additional comments or explanations beyond outputting the category label that best matches the individual's described relationship history and reasons for being single."
),
    "Looking-For": (
f"""The following are 'Looking For' category labels to help categorize what people are seeking in a future partner. These labels will be used to help match people with compatible relationship goals and preferences. Your task is to analyze the person's description of their ideal match and categorize them by outputting only the label phrase that best captures what they are looking for. Focus on the most frequently mentioned or strongly emphasized qualities.\n\n"
"Here are the labels for 'Looking For':\n\n"
"1. Attractive and Physically Fit - Seeking a partner who is physically attractive, maintains a healthy lifestyle, and enjoys being active.\n\n"
"2. Intelligent and Intellectually Curious - Values intelligence, education, and a partner who enjoys intellectual conversations and exploring different cultures.\n\n"
"3. Kind and Compassionate - Seeking a partner who is kind-hearted, empathetic, and caring towards others.\n\n"
"4. Independent and Career-Driven - Appreciates a self-sufficient, ambitious partner who is focused on their career and driven to succeed.\n\n"
"5. Adventurous and Spontaneous - Desires a partner who loves trying new things, exploring new places, and is open to excitement.\n\n"
"6. Family-Oriented and Nurturing - Wants a partner who values family, is nurturing, and is ready for family life.\n\n"
"7. Shared Interests and Lifestyle - Looking for someone with common interests, hobbies, and a compatible lifestyle.\n\n"
"8. Emotionally Mature and Stable - Values a partner who is emotionally intelligent, stable, and capable of handling life's challenges.\n\n"
"9. Strong Communicator and Supportive - Prioritizes open, honest communication and a partner who is encouraging and supportive.\n\n"
"10. Confident and Self-Assured - Appreciates a partner with high self-esteem and independence.\n\n"
"11. Loyal and Trustworthy - Seeking a partner who values commitment, honesty, and can be trusted to build a strong relationship.\n\n"
"12. Humorous and Fun-Loving - Desires a partner with a great sense of humor who enjoys having fun and maintains a playful spirit.\n\n"
"13. Cultured and Well-Rounded - Values a partner with diverse experiences, cultural understanding, and refined tastes.\n\n"
"14. Affectionate and Romantic - Prioritizes physical touch, expressions of love, and romantic gestures in a relationship.\n\n"
"15. Spiritual or Faith-Oriented - Seeking a partner who shares similar spiritual or religious beliefs and is devoted to their faith.\n\n"
"16. Social and Outgoing - Prefers someone who enjoys socializing, attending events, and meeting new people.\n\n"
"17. Domestic and Home-Oriented - Values a partner who enjoys creating a warm living environment and takes pride in homemaking.\n\n"
"18. Successful and Ambitious - Looking for a partner who is successful in their career, ambitious, and motivated to achieve their goals.\n\n"
"19. Generous and Supportive - Seeking a partner who is generous with their time, resources, and support.\n\n"
"20. Looking For Undetermined - A fallback option for when the person's description doesn't clearly fit into the other categories.\n\n"
"Analyze the input and output only the single most appropriate category label that best aligns with the person's stated preferences in a future partner. Do not offer any additional commentary or explanations."
),
    "Work-life": (
f"""The following are 'Work and Profession' category labels to help categorize what people are seeking or valuing in their professional lives. These labels will be used to help match people with compatible professional goals and preferences. Your task is to analyze the person's description of their work and professional priorities and categorize them by outputting only the label phrase that best captures their professional status or preference. Focus on the most frequently mentioned or strongly emphasized qualities.\n\n"
"Here are the labels for 'Work and Profession':\n\n"
"1. Business and Finance: Roles such as investment banker, financial advisor, wealth manager, hedge fund manager, finance executive, banker.\n\n"
"2. Technology: Roles such as software engineer, IT specialist, data scientist, technology consultant, cybersecurity analyst, tech entrepreneur.\n\n"
"3. Healthcare: Roles such as doctor, nurse, medical researcher, healthcare administrator, pharmacist, dentist.\n\n"
"4. Legal and Law: Roles such as lawyer, judge, paralegal, legal consultant, corporate lawyer.\n\n"
"5. Real Estate: Roles such as real estate agent, property developer, real estate investor, property manager.\n\n"
"6. Education: Roles such as teacher, professor, educational consultant, school administrator, tutor.\n\n"
"7. Arts and Entertainment: Roles such as actor, musician, writer, artist, film director, producer.\n\n"
"8. Sales and Marketing: Roles such as marketing manager, sales executive, brand manager, PR specialist, advertising executive.\n\n"
"9. Engineering and Manufacturing: Roles such as mechanical engineer, electrical engineer, civil engineer, chemical engineer, manufacturing specialist.\n\n"
"10. Hospitality and Service Industry: Roles such as chef, hotel manager, event planner, restaurant owner, bartender.\n\n"
"11. Entrepreneur and Self-Employed: Roles such as business owner, freelancer, consultant, self-employed, start-up founder.\n\n"
"12. Retired: Indicates a retired professional.\n\n"
"13. Work-Life Balance: Indicates the importance of work-life balance, such as work being very important, not important, or seeking work-life balance.\n\n"
"14. Wealth and Investments: Indicates wealth and financial independence, such as being wealthy with investments, having financial independence, being a trust fund beneficiary, or a high net worth individual.\n\n"
"15. Independently Wealthy: Indicates someone who is independently wealthy and does not need to work.\n\n"
"16. Just Starting Out in Life: Indicates someone who is early in their career, such as recent graduates or those just entering the workforce.\n\n"
"17. In Career Transition: Indicates someone who is changing careers or currently in a transitional phase in their professional life.\n\n"
"18. I was a Stay-at-Home Parent: Indicates someone who previously focused on parenting and managing a household, and may now be entering or re-entering the workforce.\n\n"
"19. Unable to Determine: A fallback option for when the person's description doesn't clearly fit into the other categories.\n\n"
"Analyze the input and output only the single most appropriate category label that best aligns with the person's stated professional status or preference. Do not offer any additional commentary or explanations."
),
}

def categorize_text(text, category):
    """Categorizes text using the Anthropic API."""
    if category == "All":
        results = {}
        for cat in system_prompts:
            results[cat] = categorize_text(text, cat)
        return results
    else:
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4096,
                temperature=0.4,
                system=system_prompts[category],
                messages=[
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
            # Check the type of the content and handle accordingly
            if isinstance(message.content, list):
                # Assuming each ContentBlock has a 'text' attribute
                categorized_text = ' '.join(block.text for block in message.content if hasattr(block, 'text'))
            else:
                categorized_text = message.content
            return categorized_text
        except Exception as e:
            logging.error(f"Error during categorization: {e}")
            return "Error: Unable to categorize text."

if __name__ == "__main__":
    while True:
        text = input("Enter the text you want to categorize: ")
        category = input("Enter the category (or 'All' for all categories): ")
        result = categorize_text(text, category)
        print(json.dumps(result, indent=4))
