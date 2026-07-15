import csv
import random

# We want 8760 hours
hours = 8760

with open('solar_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header
    writer.writerow(['Hour', 'Solar_Input', 'Load_Demand'])
    
    for h in range(hours):
        # A simple "Day/Night" cycle for solar (0 at night, peak at noon)
        # Using a sine wave + some random noise for "variance"
        time_of_day = h % 24
        solar = max(0, 50 * (-(time_of_day - 12)**2 / 36 + 1)) + random.uniform(-5, 5)
        
        # Load demand (higher during the day, base load at night)
        load = 10 + random.uniform(0, 5)
        
        writer.writerow([h + 1, round(solar, 2), round(load, 2)])

print("Successfully created data.csv!")