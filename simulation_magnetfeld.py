'''
=============================== SIMULATION DES MAGNETFELDES ===============================
sPam 2017 ------------------------------------------------------------- Janis, Natalie, Tim

Die Grundidee der Simulation ist, mittels einer Leiterschleife im Erdinneren (Kreisradius 
cr, Strom i) das Magnetfeld an einem gegebenen Punkt P (Längen-, Breitengrad & Höhe) zu be-
rechnen. Hierzu wird der Leiter in n Vektoren unterteilt und der Einfluss eines jedes auf 
den Punkt P berechnet. Resultat ist ein Vektor B [T]. Hieraus kann dann der Winkel zum Nor-
malvektor am Punkt P mittels angle(...) herausgefunden werden.

Nach unseren Überlegungen hat der Kreisstrom nur einen Einfluss auf die Stärke des Magnet-
feldes, der Kreisradius jedoch auf Winkel und Stärke. Die Idee der Optimierung ist, dass 
zuerst der Winkel (Kreisradius) optimiert wird und danach die Stromstärke.
===========================================================================================
'''

# Import Module & Daten
from numpy import array, pi, sin, cos, arcsin, arccos, cross, dot, sqrt
from numpy.linalg import norm
import geo

# Globale Variablen
mu0 = 4 * pi * 1e-7  # magn. Feldkonstante
re = 6.3710e6        # Erdradius

# Verwendete Pickle-Datei
pickle = 'geodata4.pickle'

# Funktion um 3d-Vektor zu erstellen
def vec3d(x, y, z):
    return array([x, y, z])

# Vektor aus Höhe, Längen- & Breitengrad
def toCart(lg, bg, h):
    lg = lg / 180 * pi  # Längengrad als rad
    bg = bg / 180 * pi  # Breitengrad als rad
    r = re + h          # Abstand vom Origo zum Punkt

    # Resultat: Kugelkoordinaten zu kartesische Koordinaten
    return vec3d(
        r * cos(bg) * cos(lg),
        r * cos(bg) * sin(lg),
        r * sin(bg)
    )

# Magnetfeld berechnen (Längengrad [°], Breitengrad [°], Höhe [m], Stromstärke [A], Kreisradius [m])
def bfield(lg, bg, h, i, cr):

    # Vektor vom Ursprung zum Punkt P (lg, bg, h)
    rp = toCart(lg, bg, h)

    # Magnetfeld als Nullvektor
    B = vec3d(0.0, 0.0, 0.0)

    n = 100            # Auflösung, wie oft Kreis unterteilt wird
    step = 2 * pi / n  # Schrittgrösse, wie gross jedes Kreisfragment ist

    # Einfluss der einzelnen Leiterschleifen-Abschnitte
    # Ein Abschnitt wird durch zwei Punkte (P1 & P2) beschrieben
    for k in range(n):

        # Vektor vom Origo zu P1
        rp1  = vec3d(
            cos(k * step), 
            sin(k * step), 
            0
        ) * cr

        # Vektor vom Origo zu P2
        rp2  = vec3d(
            cos((k + 1) * step), 
            sin((k + 1) * step), 
            0
        ) * cr

        # Vektor eines Kreisabschnitts
        rl = rp2 - rp1

        # Vektor vom Punkt P1 zum Punkt P
        rp1p = rp - rp1

        # Feldstärke auf Punkt P vom Kreisabschnitt (P1-P2) wird zum Resultat addiert
        B += mu0 * i / 4 / pi * cross(rl, rp1p) / norm(rp1p)**3

    # Rückgabe berechnetes Magnetfeld am Punkt P
    return B


# Inklinationswinkel berechnen
def angle(b, lg, bg, h):

    # Vektor zum Punkt P
    rp = toCart(lg, bg, h)

    # Winkel zwischen Normalvektor (= rp) und Magentfeld, in rad.
    phi = arccos(dot(b, rp) / norm(b) / norm(rp))

    # Winkel zum Normalvektor in Grad
    return phi / pi * 180
    
# Verarbeitung gemessener Daten
def processData(data):
    # Leere Listen worin Messdaten abgespeichert werden
    mes_lgs  = [] # Längengrade
    mes_bgs  = [] # Breitengrade
    mes_alts = [] # Höhen
    mes_mags = [] # Stärke der magn. Flussdichte
    mes_vecs = [] # B als Vektor

    # Die zu nutzenden Daten werden übernommen
    for entry in data:
        mes_lgs.append(entry[geo.ilong] / pi * 180)
        mes_bgs.append(entry[geo.ilat] / pi * 180)
        mes_alts.append(entry[geo.ialt])
        mes_mags.append(norm(entry[geo.iB]))
        mes_vecs.append(entry[geo.iB])

    # Rückgabe
    return mes_lgs, mes_bgs, mes_alts, mes_mags, mes_vecs

# Fehler zweier Vektoren berechnen
def computeErrorVector(mes_bs, sim_bs):
    # Resultat
    res = 0.0

    # Summe aller Fehler
    for k in range(len(mes_bs)):
        res += (norm(mes_bs[k] - sim_bs[k])**2) / (norm(mes_bs[k])**2)

    # Faktor & Wurzel
    return 1 / len(mes_bs) * sqrt(res)

# Funktion um den Fehler des Winkels zu berechnen (Durchschnitt)
def computeErrorNumeric(mes_ang, sim_ang):
    # Resultat
    res = 0.0

    # Summe aller Fehler
    for k in range(len(mes_ang)):
        res += ((mes_ang[k] - sim_ang[k]) / mes_ang[k])**2

    # Faktor & Wurzel
    return 1 / len(mes_ang) * sqrt(res)

'''
================================= OPTIMIERUNG DES WINKELS =================================
Erste Idee:
Anfangsradius ist anfangs klein, Fehler gross; Radius wird vergrössert (Fehler nimmt ab) bis 
der Fehler wieder zunimmt. Dann wird die Schrittgrösse verkleinert udn die Richtung geändert
bis der Fehler wieder grösser wird.
etc.
-> keine logischen Ergebnisse, siehe zweite Idee
============================================================================================
'''

# Automatische Optimierung des Kreisradius, Winkel wird genauer
def angleOptimization(resolution):
    # Gemessene Daten werden geladen
    mes_lgs, mes_bgs, mes_alts, mes_mags, mes_bs = processData(geo.load(pickle))

    # Anfangswerte
    i = 1e9
    cr = 1

    # Anfangsfehler (für Vergleich)
    last_error = 0

    # Anfangs-'Richtung' für Optimierung
    direction = True

    step_size = 2

    last_last_error = 0

    current_cr = 0
    last_cr = 0
    last_last_cr = 0.1
    last_last_last_cr = 0.1

    # Optimierungs-Schleife
    for j in range(resolution):

        if (direction):
            cr *= step_size
        else:
            cr /= step_size

        # Leere Listen für simulierte Resultate
        sim_angs = []
        mes_angs = []

        # Simulierte Werte für jedes Messdaten-Set
        for index in range(len(mes_lgs)):
            # Gemessene Werte
            mes_lg  = mes_lgs[index]  # Längengrad in rad
            mes_bg  = mes_bgs[index]  # Breitengrad in rad
            mes_alt = mes_alts[index] # Höhe
            mes_b   = mes_bs[index]   # Feld als Vektor
            mes_mag = mes_mags[index] # Strärke von B
            mes_ang = angle(mes_b, mes_lg, mes_bg, mes_alt) # Winkel zur Erdoberfläche

            # Simulierte Werte am gleichen Ort berechnen
            sim_b   = bfield(mes_lg, mes_bg, mes_alt, i, cr)
            sim_ang = angle(sim_b, mes_lg, mes_bg, mes_alt)

            # Berechnete Winkel für Fehlerrechnung ausserhalb Schleife abgespeichert
            mes_angs.append(mes_ang)
            sim_angs.append(sim_ang)

        # Fehler brechnen
        current_error = computeErrorNumeric(mes_angs, sim_angs)
        print(str(current_error) + ' ' + str(cr))

        last_last_last_last_cr = last_last_last_cr
        last_last_last_cr = last_last_cr
        last_last_cr = last_cr
        last_cr      = current_cr
        current_cr   = cr
        
        # Vergleich jetziger & vorheriger Fehler
        if (current_error > last_error):
            direction = not(direction)

        if (last_last_cr == current_cr):
            step_size /= 2
            print('now')


        # Jetziger Fehler für Vergleich später abgespeicher
        last_error = current_error


    # Rückgabe des optimierten Kreisradius
    return cr

'''
============================================================================================
Zweite Idee:
Verschiedene Radien (min & max vorgegeben, z.B. 1e6 - 6e6, n Schritte) simulieren, dann im 
Bereich mit den kleinsten Fehlern (z.B. 4e6 - 5e6) rekursiv weiterfahren. (Die Anzahl der 
Iterationen kann bestimmt werden.)
============================================================================================
'''

# Iterations-Schleife
def optimzeAngleLoop(start, end, steps, iterations):

    upper = start  # Start-Radius (Minimum)
    lower = end      # End-Radius (Maximum)
    cr, min_error = 0, 0 # Variablen für diese scope

    # Optimierungsschleife, work-in-progress
    for n in range(iterations):
        cr, min_error, upper, lower = optimzeAngle(upper, lower, steps)
        
    # Optimaler Radius mit kleinstem Fehler wird zurückgegeben
    return cr, min_error
        
# Schleife für eine Iteration
def optimzeAngle(start, end, steps):
    
    i = 1e9

    # Resultate (leere Listen)
    errors = []
    crs = []

    # Daten ab Pickle-Datei
    mes_lgs, mes_bgs, mes_alts, mes_mags, mes_bs = processData(geo.load(pickle))
    
    # Jede unterteilung (min - max, k Schritte)
    for k in range(steps):
        # Radius eines jeden Schritts, in Liste gespeichert
        cr = start + k * (end - start) / steps 
        crs.append(cr)

        # Leere Listen für simulierte Resultate
        sim_angs = []
        mes_angs = []

        # Simulierte Werte für jedes Messdaten-Set
        for index in range(len(mes_lgs)):
            # Gemessene Werte
            mes_lg  = mes_lgs[index]  # Längengrad in rad
            mes_bg  = mes_bgs[index]  # Breitengrad in rad
            mes_alt = mes_alts[index] # Höhe
            mes_b   = mes_bs[index]   # Feld als Vektor
            mes_mag = mes_mags[index] # Strärke von B
            mes_ang = angle(mes_b, mes_lg, mes_bg, mes_alt) # Winkel zum Normalvektor

            # Simulierte Werte am gleichen Ort berechnen
            sim_b   = bfield(mes_lg, mes_bg, mes_alt, i, cr)
            sim_ang = angle(sim_b, mes_lg, mes_bg, mes_alt)

            # Berechnete Winkel für Fehlerrechnung ausserhalb Schleife abgespeichert
            mes_angs.append(mes_ang)
            sim_angs.append(sim_ang)

        # Fehler brechnen
        err = computeErrorNumeric(mes_angs, sim_angs)
        errors.append(err)

    min_error    = min(errors)                      # kleister Fehler wird herausgesucht
    min_error_cr = crs[errors.index(min_error)]     # dazu gehörender Radius

    # Bereich des kleinsten Fehlers wird weitergegeben
    # Fallunterscheidung: ob die Bereiche möglich sind
    if (len(crs) - 1 < errors.index(min_error) + 1):
        upper_bound = crs[errors.index(min_error)]
        lower_bound = crs[errors.index(min_error) - 1]
    elif (errors.index(min_error) - 1 < 0):
        upper_bound = crs[errors.index(min_error) + 1]
        lower_bound = crs[errors.index(min_error)]
    else:
        upper_bound = crs[errors.index(min_error) + 1]
        lower_bound = crs[errors.index(min_error) - 1]

    # Resultate dieser Iteration
    return min_error_cr, min_error, upper_bound, lower_bound

'''
====================================== OPTIMIERUNG |B| =====================================
Die Funktionesweise ist ähnlich. Die Werte für den Strom I werden rekursiv ermittelt. Hierzu
gibt es wieder um zwei Funktionen.
============================================================================================
'''

# Iterations-Schleife
def optimizeMagnitudeLoop(start, end, steps, iterations):

    upper_bound = end   # Erster Startwert (min)
    lower_bound = start # Maximalwert für I

    # Variablen sollen in dieser scope ansprechbar sein
    min_error_i = None
    min_error = None

    # Iterationen
    for j in range(iterations):
        min_error_i, min_error, upper_bound, lower_bound = optimizeMagnitude(upper_bound, lower_bound, steps)

    # Resultate
    return min_error_i, min_error 
        
# Schleife einer Iteration
def optimizeMagnitude(start, end, steps):
    # Messdaten
    mes_lgs, mes_bgs, mes_alts, mes_mags, mes_bs = processData(geo.load(pickle))

    # Arrays für Resultate
    i_s = []
    errors = []

    # Kreisradius
    cr = 5e6

    # Berechnung von I und dessen Fehler für den Abschnitt k
    for k in range(steps):

        # I wird ermittelt und abgespeichert
        i = start + k * (end - start) / steps
        i_s.append(i)

        # Werte werden abgespeichert um Fehler ermitteln zu können
        m_mags = []
        s_mags = []

        # Simulierte Werte für jedes Messdaten-Set
        for index in range(len(mes_lgs)):
            
            # Gemessene Werte
            mes_lg  = mes_lgs[index]  # Längengrad in rad
            mes_bg  = mes_bgs[index]  # Breitengrad in rad
            mes_alt = mes_alts[index] # Höhe
            mes_mag = mes_mags[index] # Strärke von B

            # Simulierte Werte am gleichen Ort berechnen
            sim_mag = norm(bfield(mes_lg, mes_bg, mes_alt, i, cr))

            # Berechnete Winkel für Fehlerrechnung ausserhalb Schleife abgespeichert
            m_mags.append(mes_mag)
            s_mags.append(sim_mag)

        # Fehler berechnen und speichern
        err = computeErrorNumeric(m_mags, s_mags)
        errors.append(err)

    min_error = min(errors)                    # kleister Fehler wird herausgesucht
    min_error_i = i_s[errors.index(min_error)] # dazu gehörender Strom

    # Range für die nächste Iteration (Bereich worin sich der optimale Strom befinden soll)
    if (len(i_s) - 1 < errors.index(min_error) + 1):
        upper_bound = i_s[errors.index(min_error)]
        lower_bound = i_s[errors.index(min_error) - 1]
    elif (errors.index(min_error) - 1 < 0):
        upper_bound = i_s[errors.index(min_error) + 1]
        lower_bound = i_s[errors.index(min_error)]
    else:
        upper_bound = i_s[errors.index(min_error) + 1]
        lower_bound = i_s[errors.index(min_error) - 1]
    
    # Resultate werden zurückgegeben
    return min_error_i, min_error, upper_bound, lower_bound

# Hauptfunktion
def main():
    # Optionen des Programms:
    #  A) B simulieren ab Koordinaten & Höhe
    #  B) Alle Messerte anzeigen
    #  C) Winkeloptimierung durchführen
    #  D) |B| optimieren
    options_text = '  A) Rechnen \n  B) Messwerte anzeigen \n  C) Winkeloptimierung \n  D) |B| optimieren'
    print('Was soll gemacht werden? [A, B, C]')
    print(options_text)

    # Warten auf gültige Nutzereingabe
    m = True
    while m:
        a = input(':').upper()
        if (a == 'A' or a == 'B' or a == 'C' or a == 'D'):
            m = False

    # B ab einzugebenen Werten simulieren
    if (a == 'A'):
        lg  = float(input('Längengrad: \t') or 0)    # Längengrad als Kommazahl (keine Eigabe: 0)
        bg  = float(input('Breitengrad: \t') or 0)   # Breitengrad    "      " 
        h   = float(input('Höhe: \t\t') or 0)        # Höhe           "      "
        i   = float(input('Strom: \t\t') or 1e9)     # Strom (keine Eingabe: 1e9)   "       "
        cr  = float(input('Kreisradius: \t') or 5e6) # Kreisradius    "      "

        res = bfield(lg, bg, h, i, cr) # Feld als Vektor brechnen
        ang = angle(res, lg, bg, h)    # Winkel zum normalvektor

        # Anzeigen der Resultate
        print(' B \t' + str(res) + '\n|B| \t' + str(norm(res)) + '\n Angle \t' + str(90 - ang))

    # Alle Messdaten anzeigen
    elif (a == 'B'):
        # Daten laden
        mes_lgs, mes_bgs, mes_alts, mes_mags, mes_bs = processData(geo.load(pickle))
        # Alle Daten anzeigen
        print('lg \tbg \th \tmag \t\tang \tB')
        for k in range(len(mes_lgs)):
            print(
                str(round(mes_lgs[k], 2)) + '\t' +
                str(round(mes_bgs[k], 2)) + '\t' +
                str(int(mes_alts[k])) + '\t' +
                str(round(mes_mags[k], 8)) + '\t' +
                str(round(angle(mes_bs[k], mes_lgs[k], mes_bgs[k], mes_alts[k]), 3)) + '\t' +
                str(mes_bs[k]) 
            )

    # Winkeloptimierung mittels Kreisradiusänderung
    elif (a == 'C'):
        # Eingabe: Schrittgrösse, Iterationen, Start-/Endwert, Kreisradius
        steps      = int(input("Anz. Schritte: \t"))
        iterations = int(input("Interationen: \t"))
        start      = int(input("Anfangswert: \t"))
        end        = int(input("Endwert: \t"))

        # optimaler Radius & Fehler ermitteln
        cr, err = optimzeAngleLoop(start, end, steps, iterations)
        
        # Werte anzeigen
        print('optimaler Kreisradius: \t' + str(cr) + '\nmit Fehler: \t' + str(err))

    # Optimierung |B| durch Stromstärke
    else:
        # Eingabe: Schrittgrösse, Iterationen, Start-/Endwert
        steps      = int(input("Anz. Schritte: "))
        iterations = int(input("Interationen: "))
        start      = int(input("Anfangswert: "))
        end        = int(input("Endwert: "))

        # optimaler Strom mit Fehler ermitteln
        i, err = optimizeMagnitudeLoop(start, end, steps, iterations)
    
        # Werte Anzeigen
        print('optimaler Strom: \t' + str(i) + '\nmit Fehler: \t' + str(err))

# Programmstart
main()