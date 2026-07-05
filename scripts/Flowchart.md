START
  |
  v
User selects or enters fishing request
  |
  v
System reads selected species, month, year, and map location
  |
  v
Is selected species available in historical data?
  |
  +-- NO --> Recommend: "Not enough expert data for this species"
  |
  +-- YESw
        |
        v
    Is the selected month in the species' active season?
        |
        +-- YES --> Add rule: "Season match" (+ve certainty)
        |
        +-- NO --> Add rule: "Season mismatch" (-ve certainty)
        |
        v
    Has this area had high historical catch density?
        |
        +-- YES --> Add rule: "High catch area" (+ve certainty)
        |
        +-- NO --> Add rule: "Low catch area" (-ve certainty)
        |
        v
    Is the location inside the common latitude/longitude range for this species?
        |
        +-- YES --> Add rule: "Location match" (+ve certainty)
        |
        +-- NO --> Add rule: "Location outside normal range" (-ve certainty)
        |
        v
    Does the user have positive feedback for similar requests?
        |
        +-- YES --> Add rule: "User preference match" (+ve certainty)
        |
        +-- NO --> No preference bonus
        |
        v
    Calculate final certainty score
        |
        v
    IF certainty is high
        THEN recommend the spot strongly
    ELSE IF certainty is medium
        THEN recommend with caution
    ELSE
        THEN suggest changing species, month, or location
        |
        v
END