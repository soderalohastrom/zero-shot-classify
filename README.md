# Person Profle API 
**FastAPI, Docker, and Hugging Face Transformer**
```
CATEGORIES = {
    "Politics": [
        "Republican Family Traditionalists",
        "God and Guns Conservatives",
        "Progressive Activist",
        "Loyal Democratic Voter",
        "Moderate Pragmatist",
        "Libertarian Thinker",
        "Apolitical Minimalist",
        "Eco-Conscious Liberal",
        "Independent Swing Voter",
        "Cultural Conservative",
        "Liberal Intellectual",
        "Non-Political Humanist",
        "Fiscally Conservative, Socially Liberal",
        "Politically Flexible Pragmatist",
        "Trump Followers",
        "Politics Undetermined"  # Fallback option
    ],    
    "Lifestyle": [
        "Adventurous Traveler",
        "Health Enthusiasts",
        "Social Butterflies",
        "Fitness Fanatics",
        "Culinary Explorers",
        "Family and Community Oriented",
        "Intellectuals and Lifelong Learners",
        "Balanced Work-Life Advocates",
        "Homebodies with Hobbies",
        "Active and Outdoorsy",
        "Spiritual and Mindful",
        "Cultural Connoisseurs",
        "Wellness and Self-Care Advocates",
        "Social and Professional Networkers",
        "Community Builders and Volunteers",
        "Nature Lovers and Environmentalists",
        "Simplifiers and Decluttering Enthusiasts",
        "Philanthropists and Humanitarians",
        "Pet Owners and Animal Lovers",
        "Adrenaline Junkies and Thrill Seekers",
        "Sustainable Living Advocates",
        "Career Achievers and Go-Getters",
        "Luxury and High-End Lifestyle Enthusiasts",
        "Free Spirits and Nonconformists",
        "Classic and Timeless Style Aficionados",
        "Country Living and Rustic Enthusiasts",
        "City Dwellers and Urban Explorers",
        "Digital Nomads and Remote Workers",
        "Lifestyle Undetermined"  # Fallback option
    ],
    "Children": [
        "No Children, Wants Children",
        "No Children, Open to Having Children",
        "No Children, Does Not Want Children",
        "Has Children, Open to More",
        "Has Children, Does Not Want More",
        "No Preference for Having Children",
        "Specific Conditions for Having/Accepting Children",
        "Has Children, Specific Preferences for New Children",
        "Adoptive Parents",
        "No Children, Prefers Not to Have Children",
        "No Children, Unsure About Having Children",
        "Has Adult Children, No Interest in Having More",
        "No Children, Prefers No Kids",
        "Child Preference Undetermined"  # Fallback option
    ],
    "Upbringing": [
        "Traditional Family Structure",
        "Single Parent Household",
        "Blended Family",
        "Adoptive or Foster Family",
        "Multicultural Background",
        "Religious Upbringing",
        "Migratory Upbringing",
        "Rural Upbringing",
        "Inner-City Upbringing", 
        "Urban Upbringing",
        "Rural, then Urban",
        "Suburban Upbringing",
        "Suburban - Unconventional"
        "Military Family Background",
        "Immigrant Family Background",
        "Academic-Focused Upbringing",
        "Entrepreneurial Family",
        "Bohemian - Communal Upbringing",
        "Artistic or Creative Family",
        "Upbringing Undetermined"  # Fallback option   
    ],
    "Geo-Familiarity": [
        "California",
        "Florida",
        "Texas & Deep South",
        "NYC & DC-to-Boston Corridor",
        "Northern New England",
        "Atlantic Coast & Appalachia",
        "Great Lakes & Midwest",
        "Mountain West and Prairie",
        "Southwest Desert",
        "Cascadia & Alaska",
        "Canada",
        "Mexico & South America",
        "Western Europe",
        "Middle East & North Africa",
        "Australia & Pacific Island",
        "Tropical Africa & Caribbean",
        "Asia & The Baltics",
        "Region Undetermined"  # Fallback option
    ],
    "Why-Single": [
        "Divorced - Married too young",
        "Divorced - Long marriage, grew apart",
        "Divorced - Infidelity",
        "Divorced - Incompatibility",
        "Divorced - Spouse's personal issues",
        "Divorced - Disagreements over having children",
        "Divorced - Career or lifestyle differences",
        "Divorced - Lack of emotional intimacy or connection",
        "Divorced - Other reasons",
        "Widowed - Ready to find love again",
        "Never married - Focused on career or personal growth",
        "Never married - Hasn't found the right person",
        "Long-term relationship ended - Incompatibility",
        "Long-term relationship ended - Distance or logistics",
        "Long-term relationship ended - Commitment issues",
        "Long-term relationship ended - Other reasons",
        "Dating casually - Seeking the right connection",
        "Relationship history varied - Open to new possibilities",
        "Prefers not to say",
        "Relationship History Undetermined"  # Fallback option
    ],
      "Looking-For": [
        "Attractive and Physically Fit",
        "Intelligent and Intellectually Curious",
        "Kind and Compassionate",
        "Independent and Career-Driven",
        "Adventurous and Spontaneous",
        "Family-Oriented and Nurturing",
        "Shared Interests and Lifestyle",
        "Emotionally Mature and Stable",
        "Strong Communicator and Supportive",
        "Confident and Self-Assured",
        "Loyal and Trustworthy",
        "Humorous and Fun-Loving",
        "Cultured and Well-Rounded",
        "Affectionate and Romantic",
        "Spiritual or Faith-Oriented",
        "Social and Outgoing",
        "Domestic and Home-Oriented",
        "Successful and Ambitious",
        "Generous and Supportive",
        "Looking For Undetermined"
    ],
        "Work-life": [
        "Business and Finance",
        "Technology",
        "Healthcare",
        "Legal and Law",
        "Real Estate",
        "Education",
        "Arts and Entertainment",
        "Sales and Marketing",
        "Engineering and Manufacturing",
        "Hospitality and Service Industry",
        "Entrepreneur and Self-Employed",
        "Retired",
        "Work-Life Balance",
        "Wealth and Investments",
        "Independently Wealthy",
        "Unable to Determine"
    ]
}
```

