IF species exists in the historical dataset
THEN continue expert analysis
ELSE return "insufficient data"

IF selected month is commonly associated with that species
THEN increase certainty factor by 25

IF selected map area has high historical catch density
THEN increase certainty factor by 35

IF selected location is within the species' normal latitude and longitude range
THEN increase certainty factor by 25

IF the user has previously rated similar searches positively
THEN increase certainty factor by 15

IF total certainty factor >= 75
THEN give a strong fishing recommendation

IF total certainty factor is between 45 and 74
THEN give a moderate recommendation

IF total certainty factor < 45
THEN advise the user to change species, season, or location

