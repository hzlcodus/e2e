import pandas as pd

# Your data
data = [
    {"name": "Ss Crew Neck T-Shirt - Black", "season": "SS23", "brand": "beaker", "color": "Black", "material": "코튼, 데님", "style": "티셔츠", "fit": "레귤러핏"},
    {"name": "❄️시즌오프❄️ Bulletproof Jumper - Charcoal", "season": "FW23", "brand": "beaker", "color": "Charcoal", "button_zipper": "투웨이 지퍼", "style": "점퍼"},
    {"name": "❄️시즌오프❄️ Men Black Watch Cotton Shorts - Green", "season": "FW23", "brand": "beaker", "color": "Green"},
    {"name": "❄️시즌오프❄️ ★강민혁 착용★ Overshirt Desco - Navy", "season": "FW23", "brand": "beaker", "color": "Navy", "material": "코튼", "style": "셔츠"},
    {"name": "❄️시즌오프❄️ Padding Sleeve Sweatshirts - Black", "season": "FW23", "brand": "beaker", "color": "Black", "material": "면", "style": "셔츠"},
    {"name": "Stand Collar Blouson - Stone", "season": "FW23", "brand": "beaker", "color": "Stone", "material": "코튼", "neck_detail": "칼라", "style": "블루종"},
    {"name": "Women Hard Working Hat - Blue", "season": "FW23", "brand": "beaker", "color": "Blue", "pattern/print": "스트라이프"}
]

# Create a DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel("output.xlsx", index=False)
