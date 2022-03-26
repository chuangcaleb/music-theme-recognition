all_midi_features_list = [
    'Basic Pitch Histogram',
    'Pitch Class Histogram',
    'Folded Fifths Pitch Class Histogram',
    'Number of Pitches',
    'Number of Pitch Classes',
    'Number of Common Pitches',
    'Number of Common Pitch Classes',
    'Range',
    'Importance of Bass Register',
    'Importance of Middle Register',
    'Importance of High Register',
    'Dominant Spread',
    'Strong Tonal Centres',
    'Mean Pitch',
    'Mean Pitch Class',
    'Most Common Pitch',
    'Most Common Pitch Class',
    'Prevalence of Most Common Pitch',
    'Prevalence of Most Common Pitch Class',
    'Relative Prevalence of Top Pitches',
    'Relative Prevalence of Top Pitch Classes',
    'Interval Between Most Prevalent Pitches',
    'Interval Between Most Prevalent Pitch Classes',
    'Pitch Variability',
    'Pitch Class Variability',
    'Pitch Class Variability After Folding',
    'Pitch Skewness',
    'Pitch Class Skewness',
    'Pitch Class Skewness After Folding',
    'Pitch Kurtosis',
    'Pitch Class Kurtosis',
    'Pitch Class Kurtosis After Folding',
    'Major or Minor',
    'First Pitch',
    'First Pitch Class',
    'Last Pitch',
    'Last Pitch Class',
    'Glissando Prevalence',
    'Average Range of Glissandos',
    'Vibrato Prevalence',
    'Microtone Prevalence',
    'Melodic Interval Histogram',
    'Most Common Melodic Interval',
    'Mean Melodic Interval',
    'Number of Common Melodic Intervals',
    'Distance Between Most Prevalent Melodic Intervals',
    'Prevalence of Most Common Melodic Interval',
    'Relative Prevalence of Most Common Melodic Intervals',
    'Amount of Arpeggiation',
    'Repeated Notes',
    'Chromatic Motion',
    'Stepwise Motion',
    'Melodic Thirds',
    'Melodic Perfect Fourths',
    'Melodic Tritones',
    'Melodic Perfect Fifths',
    'Melodic Sixths',
    'Melodic Sevenths',
    'Melodic Octaves',
    'Melodic Large Intervals',
    'Minor Major Melodic Third Ratio',
    'Melodic Embellishments',
    'Direction of Melodic Motion',
    'Average Length of Melodic Arcs',
    'Average Interval Spanned by Melodic Arcs',
    'Melodic Pitch Variety',
    'Vertical Interval Histogram',
    'Wrapped Vertical Interval Histogram',
    'Chord Type Histogram',
    'Average Number of Simultaneous Pitch Classes',
    'Variability of Number of Simultaneous Pitch Classes',
    'Average Number of Simultaneous Pitches',
    'Variability of Number of Simultaneous Pitches',
    'Most Common Vertical Interval',
    'Second Most Common Vertical Interval',
    'Distance Between Two Most Common Vertical Intervals',
    'Prevalence of Most Common Vertical Interval',
    'Prevalence of Second Most Common Vertical Interval',
    'Prevalence Ratio of Two Most Common Vertical Intervals',
    'Vertical Unisons',
    'Vertical Minor Seconds',
    'Vertical Thirds',
    'Vertical Tritones',
    'Vertical Perfect Fourths',
    'Vertical Perfect Fifths',
    'Vertical Sixths',
    'Vertical Sevenths',
    'Vertical Octaves',
    'Perfect Vertical Intervals',
    'Vertical Dissonance Ratio',
    'Vertical Minor Third Prevalence',
    'Vertical Major Third Prevalence',
    'Chord Duration',
    'Partial Chords',
    'Standard Triads',
    'Diminished and Augmented Triads',
    'Dominant Seventh Chords',
    'Seventh Chords',
    'Non-Standard Chords',
    'Complex Chords',
    'Minor Major Triad Ratio',
    'Initial Time Signature',
    'Simple Initial Meter',
    'Compound Initial Meter',
    'Complex Initial Meter',
    'Duple Initial Meter',
    'Triple Initial Meter',
    'Quadruple Initial Meter',
    'Metrical Diversity',
    'Total Number of Notes',
    'Note Density per Quarter Note',
    'Note Density per Quarter Note per Voice',
    'Note Density per Quarter Note Variability',
    'Rhythmic Value Histogram',
    'Range of Rhythmic Values',
    'Number of Different Rhythmic Values Present',
    'Number of Common Rhythmic Values Present',
    'Prevalence of Very Short Rhythmic Values',
    'Prevalence of Short Rhythmic Values',
    'Prevalence of Medium Rhythmic Values',
    'Prevalence of Long Rhythmic Values',
    'Prevalence of Very Long Rhythmic Values',
    'Prevalence of Dotted Notes',
    'Shortest Rhythmic Value',
    'Longest Rhythmic Value',
    'Mean Rhythmic Value',
    'Most Common Rhythmic Value',
    'Prevalence of Most Common Rhythmic Value',
    'Relative Prevalence of Most Common Rhythmic Values',
    'Difference Between Most Common Rhythmic Values',
    'Rhythmic Value Variability',
    'Rhythmic Value Skewness',
    'Rhythmic Value Kurtosis',
    'Rhythmic Value Median Run Lengths Histogram',
    'Mean Rhythmic Value Run Length',
    'Median Rhythmic Value Run Length',
    'Variability in Rhythmic Value Run Lengths',
    'Rhythmic Value Variability in Run Lengths Histogram',
    'Mean Rhythmic Value Offset',
    'Median Rhythmic Value Offset',
    'Variability of Rhythmic Value Offsets',
    'Complete Rests Fraction',
    'Partial Rests Fraction',
    'Average Rest Fraction Across Voices',
    'Longest Complete Rest',
    'Longest Partial Rest',
    'Mean Complete Rest Duration',
    'Mean Partial Rest Duration',
    'Median Complete Rest Duration',
    'Median Partial Rest Duration',
    'Variability of Complete Rest Durations',
    'Variability of Partial Rest Durations',
    'Variability Across Voices of Combined Rests',
    'Beat Histogram Tempo Standardized',
    'Number of Strong Rhythmic Pulses - Tempo Standardized',
    'Number of Moderate Rhythmic Pulses - Tempo Standardized',
    'Number of Relatively Strong Rhythmic Pulses - Tempo Standardized',
    'Strongest Rhythmic Pulse - Tempo Standardized',
    'Second Strongest Rhythmic Pulse - Tempo Standardized',
    'Harmonicity of Two Strongest Rhythmic Pulses - Tempo Standardized',
    'Strength of Strongest Rhythmic Pulse - Tempo Standardized',
    'Strength of Second Strongest Rhythmic Pulse - Tempo Standardized',
    'Strength Ratio of Two Strongest Rhythmic Pulses - Tempo Standardized',
    'Combined Strength of Two Strongest Rhythmic Pulses - Tempo Standardized',
    'Rhythmic Variability - Tempo Standardized',
    'Rhythmic Looseness - Tempo Standardized',
    'Polyrhythms - Tempo Standardized',
    'Initial Tempo',
    'Mean Tempo',
    'Tempo Variability',
    'Duration in Seconds',
    'Note Density',
    'Note Density Variability',
    'Average Time Between Attacks',
    'Average Time Between Attacks for Each Voice',
    'Variability of Time Between Attacks',
    'Average Variability of Time Between Attacks for Each Voice',
    'Minimum Note Duration',
    'Maximum Note Duration',
    'Average Note Duration',
    'Variability of Note Durations',
    'Amount of Staccato',
    'Beat Histogram',
    'Number of Strong Rhythmic Pulses',
    'Number of Moderate Rhythmic Pulses',
    'Number of Relatively Strong Rhythmic Pulses',
    'Strongest Rhythmic Pulse',
    'Second Strongest Rhythmic Pulse',
    'Harmonicity of Two Strongest Rhythmic Pulses',
    'Strength of Strongest Rhythmic Pulse',
    'Strength of Second Strongest Rhythmic Pulse',
    'Strength Ratio of Two Strongest Rhythmic Pulses',
    'Combined Strength of Two Strongest Rhythmic Pulses',
    'Rhythmic Variability',
    'Rhythmic Looseness',
    'Polyrhythms',
    'Pitched Instruments Present',
    'Unpitched Instruments Present',
    'Note Prevalence of Pitched Instruments',
    'Note Prevalence of Unpitched Instruments',
    'Time Prevalence of Pitched Instruments',
    'Variability of Note Prevalence of Pitched Instruments',
    'Variability of Note Prevalence of Unpitched Instruments',
    'Number of Pitched Instruments',
    'Number of Unpitched Instruments',
    'Unpitched Percussion Instrument Prevalence',
    'String Keyboard Prevalence',
    'Acoustic Guitar Prevalence',
    'Electric Guitar Prevalence',
    'Violin Prevalence',
    'Saxophone Prevalence',
    'Brass Prevalence',
    'Woodwinds Prevalence',
    'Orchestral Strings Prevalence',
    'String Ensemble Prevalence',
    'Electric Instrument Prevalence',
    'Maximum Number of Independent Voices',
    'Average Number of Independent Voices',
    'Variability of Number of Independent Voices',
    'Voice Equality - Number of Notes',
    'Voice Equality - Note Duration',
    'Voice Equality - Dynamics',
    'Voice Equality - Melodic Leaps',
    'Voice Equality - Range',
    'Importance of Loudest Voice',
    'Relative Range of Loudest Voice',
    'Relative Range Isolation of Loudest Voice',
    'Relative Range of Highest Line',
    'Relative Note Density of Highest Line',
    'Relative Note Durations of Lowest Line',
    'Relative Size of Melodic Intervals in Lowest Line',
    'Voice Overlap',
    'Voice Separation',
    'Variability of Voice Separation',
    'Parallel Motion',
    'Similar Motion',
    'Contrary Motion',
    'Oblique Motion',
    'Parallel Fifths',
    'Parallel Octaves',
    'Dynamic Range',
    'Variation of Dynamics',
    'Variation of Dynamics In Each Voice',
    'Average Note to Note Change in Dynamics'
]
