# add batik-mozart and vienna corpus into metadata.csv
import pickle
import csv
from pathlib import Path
import pandas as pd
import os

vienna_path = str(Path('../vienna4x22'))
batik_path = str(Path('../batik_plays_mozart'))
feature_folder = str(Path('../features'))
atepp_path = str(Path('../ATEPP'))

metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))
metadata = metadata[metadata['source'] == 'ASAP']
metadata.reset_index(inplace=True)

# ------------------ walk through the Batik Folder ---------------------
performance_id = 'Placeholder'
midi_perfm = 'Placeholder'
xml_path = 'Placeholder'
i = 0 # counter to track number of files, %10==0 will result in file being validation set

# walk through batik dataset
for (dirpath, dirnames, filenames) in os.walk(batik_path):
    for filename in filenames:
        if filename.endswith('.mid'): 
            # find corresponding musicxml score file
            performance_id = filename.split('_')[0]
            for (dirpath2, dirnames2, filenames2) in os.walk(batik_path):
                for filename2 in filenames2:
                    if filename2.endswith('.musicxml') and Path(filename2).stem == Path(filename).stem:
                        xml_path = str(Path(batik_path, 'scores/' + filename2))
                        midi_perfm = str(Path(batik_path, 'midi/' + filename))

            feature_file = Path(feature_folder, '{}.pkl'.format(performance_id))
            piece_id = performance_id
            source = 'Batik'

            if i % 10 != 0:
                split = 'train'
            else:
                split = 'valid'

            # update metadata file with new dataset information
            metadata = metadata.append({
                            'performance_id': performance_id,
                            'piece_id': piece_id,
                            'source': source,
                            'split': split,
                            'midi_perfm': midi_perfm,
                            'feature_file': str(feature_file),
                            'performance_MIDI_external': '-',
                            'xml_file': xml_path,
                        }, ignore_index=True)

            i += 1

# ------------------ walk through the vienna4x22 Folder ---------------------
# we will create custom IDs for the vienna corpus: V_XX
j = 1 # for vienna corpus id

for (dirpath, dirnames, filenames) in os.walk(vienna_path):
    for filename in filenames:
        if filename.endswith('.mid'): 
            # find corresponding musicxml score file
            performance_id = 'V_' + str(j)
            for (dirpath2, dirnames2, filenames2) in os.walk(vienna_path):
                for filename2 in filenames2:
                    if filename2.endswith('.musicxml') and str(Path(filename2).stem) == str(Path(filename).stem)[:-4]: # removing last 4 char of midi file to match piece with musicxml file
                        xml_path = str(Path(vienna_path, 'musicxml/' + filename2))
                        midi_perfm = str(Path(vienna_path, 'midi/' + filename))

            feature_file = Path(feature_folder, '{}.pkl'.format(performance_id))
            piece_id = performance_id
            source = 'Vienna'

            if i % 10 != 0:
                split = 'train'
            else:
                split = 'valid'

            # update metadata file with new dataset information
            metadata = metadata.append({
                            'performance_id': performance_id,
                            'piece_id': piece_id,
                            'source': source,
                            'split': split,
                            'midi_perfm': midi_perfm,
                            'feature_file': str(feature_file),
                            'performance_MIDI_external': '-',
                            'xml_file': xml_path,
                        }, ignore_index=True)

            i += 1
            j += 1

# ---------------------------- add ATEPP dataset ----------------------
# load from ATEPP metadata.csv
atepp_metadata = pd.read_csv(str(Path(atepp_path, 'ATEPP-metadata-1.2.csv')))

# get those pieces with an xml file
atepp_metadata = atepp_metadata[atepp_metadata['score_path'].str.len() > 0]

# from observation, ignore all Beethoven and Schumann pieces as they overlap with ASAP significantly
atepp_metadata = atepp_metadata[atepp_metadata['composer'] != 'Robert Schumann']
atepp_metadata = atepp_metadata[atepp_metadata['composer'] != 'Ludwig van Beethoven']

# handpicken files to NOT include as they already overlap with existing pieces
to_ignore = ["""Sergei_Rachmaninoff/13_Preludes,_Op._32/No._10_in_B_Minor:_Lento/musicxml_cleaned.musicxml""","""Sergei_Rachmaninoff/13_Preludes,_Op._32/No._5_in_G_Major:_Moderato/musicxml_cleaned.musicxml""","""Franz_Liszt/Années_de_pèlerinage,_2nd_Year,_S._162_"Italy_Supplement"/No._1._Gondoliera_(Gondolier's_Song)/musicxml_cleaned.musicxml""","""Franz_Liszt/2_Etudes_de_Concert,_S.145/No.2_Gnomenreigen/musicxml_cleaned.musicxml""","""Franz_Liszt/12_Etudes_d'exécution_transcendante,_S.139/No.11_Harmonies_du_soir_(Andantino)/musicxml_cleaned.musicxml""","""Franz_Liszt/12_Etudes_d'exécution_transcendante,_S.139/No.1_Prélude_(Presto)/musicxml_cleaned.musicxml""","""Franz_Liszt/12_Etudes_d'exécution_transcendante,_S.139/No.10_Allegro_agitato_molto/musicxml_cleaned.musicxml""","""Franz_Liszt/12_Etudes_d'exécution_transcendante,_S.139/No._5_Feux_follets_(Allegretto)/musicxml_cleaned.musicxml""","""Franz_Liszt/12_Etudes_d'exécution_transcendante,_S.139/No.3_Paysage_(Poco_adagio)/musicxml_cleaned.musicxml""","""Alexander_Scriabin/12_Etudes,_Op._8/No._11_in_B-Flat_Minor/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._9_in_E_Major,_BWV_854:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._12_in_F_Minor,_BWV_857:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._17_in_A-Flat_Major,_BWV_862:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._17_in_A-Flat_Major,_BWV_862:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._11_in_F_Major,_BWV_856:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._11_in_F_Major,_BWV_856:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._23_in_B_Major,_BWV_868:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._15_in_G_Major,_BWV_860:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._1_in_C_Major,_BWV_846:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._18_in_G-Sharp_minor,_BWV_863:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._22_in_B-Flat_Minor,_BWV_867:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._22_in_B-Flat_Minor,_BWV_867:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._23_in_B_Major,_BWV_868:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._19_in_A_Major,_BWV_864:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._13_in_F-Sharp_Major,_BWV_858:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._13_in_F-Sharp_Major,_BWV_858:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._20_in_A_Minor,_BWV_865:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._15_in_G_Major,_BWV_860:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._21_in_B-Flat_Major,_BWV_866:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._9_in_E_Major,_BWV_854:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._1_in_C_Major,_BWV_846:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._21_in_B-Flat_Major,_BWV_866:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._12_in_F_Minor,_BWV_857:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._18_in_G-Sharp_minor,_BWV_863:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._3_in_C-Sharp_Major,_BWV_848:_Prelude/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._19_in_A_Major,_BWV_864:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._3_in_C-Sharp_Major,_BWV_848:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/The_Well-Tempered_Clavier,_Book_I/No._20_in_A_Minor,_BWV_865:_Fugue/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_C_major_BWV_870/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_C_sharp_minor_BWV_873/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_G_major_BWV_884/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_A_minor_BWV_889/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_D_minor_BWV_875/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_A_major_BWV_888/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_A_major_BWV_888/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_G_minor_BWV_885/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_G_Sharp_Minor,_BWV_887/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_F_major_BWV_880/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_E_flat_major_BWV_876/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_B_flat_minor_BWV_891/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_B_flat_minor_BWV_891/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_B_minor_BWV_893/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_D_major_BWV_874/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_C_sharp_minor_BWV_873/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_F_sharp_minor_BWV_883/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_F_sharp_minor_BWV_883/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_D_minor_BWV_875/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_B_minor_BWV_893/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_B_major_BWV_892/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_B_major_BWV_892/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_C_major_BWV_870/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_D_major_BWV_874/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_E_flat_major_BWV_876/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_G_major_BWV_884/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Fugue_in_F_major_BWV_880/musicxml_cleaned.musicxml""","""Johann_Sebastian_Bach/Das_Wohltemperierte_Klavier_Book2/Book_2,_BWV_870-893:_Prelude_in_A_minor_BWV_889/musicxml_cleaned.musicxml""","""Claude_Debussy/Images,_Book_1,_L._110/No._1,_Reflets_dans_l'eau/musicxml_cleaned.musicxml""","""Maurice_Ravel/Miroirs,_M._43/No._3,_Une_barque_sur_l'océan/musicxml_cleaned.musicxml""","""Maurice_Ravel/Miroirs,_M._43/No._4,_Alborada_del_gracioso/musicxml_cleaned.musicxml""","""Maurice_Ravel/Gaspard_de_la_nuit,_M.55/Ondine/musicxml_cleaned.musicxml""","""Franz_Schubert/6_Moments_musicaux,_Op.94_D.780/No.3_in_F_minor_(Allegro_moderato)/musicxml_cleaned.musicxml""","""Franz_Schubert/6_Moments_musicaux,_Op.94_D.780/No.1_in_C_(Moderato)/musicxml_cleaned.musicxml""","""Franz_Schubert/Piano_Sonata_No.13_in_A,_D.664/1._Allegro_moderato/Schubert_-_Sonata_in_A_Op.120_D.664_Movement_I.mxl""","""Franz_Schubert/Piano_Sonata_No.13_in_A,_D.664/3._Allegro/musicxml_cleaned.musicxml""","""Franz_Schubert/Piano_Sonata_No.13_in_A,_D.664/2._Andante/musicxml_cleaned.musicxml""","""Franz_Schubert/Piano_Sonata_No.18_in_G,_D.894/2._Andante/musicxml_cleaned.musicxml""","""Franz_Joseph_Haydn/Piano_Sonata_in_A_Flat_Major,_Hob._XVI:46/1._Allegro_moderato/Allegro_Moderato_First_Mvt._Sonata_in_Ab_Hob_XVI_46.mxl""","""Franz_Joseph_Haydn/Piano_Sonata_in_C_Major,_Hob.XVI:48/48:_1._Andante_con_espressione/musicxml_cleaned.musicxml""","""Franz_Joseph_Haydn/Piano_Sonata_in_C_Major,_Hob.XVI:48/48:_2._Rondo_(Presto)/musicxml_cleaned.musicxml""","""Franz_Joseph_Haydn/Piano_Sonata_in_C_Major,_Hob.XVI:50/I._Allegro/musicxml_cleaned.musicxml""","""Franz_Joseph_Haydn/Piano_Sonata_in_B_Minor,_Hob.XVI:32/32:_1._Allegro_moderato/musicxml_cleaned.musicxml""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.18_in_D,_K.576/1._Allegro/Sonata_No._18_Mvt_1_Mozart.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.18_in_D,_K.576/3._Allegretto/Sonata_No._18_The_Hunt_3rd_Movement_K._576.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._18_in_D,_K.576/2._Adagio/Sonata_No._18_Mvt_2_Mozart.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.12_in_F,_K.332/1._Allegro/musicxml_cleaned.musicxml""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.12_in_F,_K.332/3._Allegro_assai/musicxml_cleaned.musicxml""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.12_in_F,_K.332/2._Adagio/musicxml_cleaned.musicxml""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._3_in_B_flat,_K.281/2._Andante_amoroso/Piano_Sonata_No.3_in_B-flat_major_K.281189f__Wolfgang_Amadeus_Mozart.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._3_in_B_flat,_K.281/3._Rondeau_(Allegro)/Sonata_No._3_3rd_Movement_K._281_Rondo.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._3_in_B_flat,_K.281/1._Allegro/Sonata_No._3_1st_Movement_K._281.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.13_in_B_flat,_K.333/1._Allegro/Sonata_No._13_Mvt_1_Mozart.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.13_in_B_flat,_K.333/3._Allegretto_grazioso/Sonata_No._13_3rd_Movement_K._333.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.13_in_B_flat,_K.333/2._Andante_cantabile/Sonata_No._13_2nd_Movement_K._333.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._5_in_G_Major,_K._283/I._Allegro/Sonata_No._5_1st_Movement_K._283.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._5_in_G_Major,_K._283/III._Presto/Sonata_No._5_3rd_Movement_K._283.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._5_in_G_Major,_K._283/II._Andante/Sonata_No._5_2nd_Movement_K._283.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.9_in_D,_K.311/1._Allegro_con_spirito/Mozart_-_Sonata_in_D_K_311_Movement_I.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.9_in_D,_K.311/3._Rondeau_(Allegro)/Sonata_No._9_3rd_Movement_K._311.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.9_in_D,_K.311/2._Andantino_con_espressione/Sonata_No._9_2nd_Movement_K._311.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._14_in_C_Minor,_K._457/III._Allegro_assai/Sonata_No._14_3rd_Movement_K._457.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._14_in_C_Minor,_K._457/I._Molto_allegro/Mozart_-_Piano_Sonata_in_C_minor_K_457_No._14_1st_movement.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._14_in_C_Minor,_K._457/II._Adagio/Sonata_No._14_2nd_Movement_K._457.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._1_in_C,_K.279/3._Allegro/Sonata_No._1_3rd_Movement_K._279.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._1_in_C,_K.279/1._Allegro/Sonata_No._1_1st_Movement_K._279.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._1_in_C,_K.279/2._Andante/Sonata_No._1_2nd_Movement_K._279.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._17_in_B-Flat_Major,_K._570/I._Allegro/Mozart_-_Sonata_in_Bb_K_570_Movement_I.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._17_in_B-Flat_Major,_K._570/II._Adagio/Sonata_No._17_2nd_Movement_K._570.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._17_in_B-Flat_Major,_K._570/III._Allegretto/Sonata_No._17_3rd_Movement_K._570.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.4_in_E_flat,_K.282/3._Allegro/Sonata_No._4_3rd_Movement_K._282.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.4_in_E_flat,_K.282/1._Adagio/Sonata_No._4_1st_Movement_K._282.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.4_in_E_flat,_K.282/2._Menuetto_I-II/Sonata_No._4_2nd_Movement_K._282.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._11_in_A_Major,_K._331/2._Menuetto/Sonata_No._11_2nd_Movement_K._331.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._11_in_A_Major,_K._331/1._Tema_(Andante_grazioso)_con_variazioni/Sonata_No._11_1st_Movement_K._331.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._11_in_A_Major,_K._331/3._Alla_Turca._Allegretto/musicxml_cleaned.musicxml""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._10_In_C_Major,_K.330/1._Allegro_moderato/Sonata_No._10_1st_Movement_K._330.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._10_In_C_Major,_K.330/3._Allegretto/Sonata_No._10_3rd_Movement_K._330.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._10_In_C_Major,_K.330/2._Andante_cantabile/Sonata_No._10_2nd_Movement_K._330.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.8_in_A_minor,_K.310/3._Presto/Sonata_No._8_3rd_Movement_K._310.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.8_in_A_minor,_K.310/2._Andante_cantabile_con_espressione/Sonata_No._8_2nd_Movement_K._310.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.8_in_A_minor,_K.310/1._Allegro_maestoso/musicxml_cleaned.musicxml""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._2_in_F_Major,_K._280/II._Adagio/Sonata_No._2_2nd_Movement_K._280.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._2_in_F_Major,_K._280/III._Presto/Sonata_No._2_2nd_Movement_K._280.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._2_in_F_Major,_K._280/I._Allegro_assai/Sonata_No._2_1st_Movement_K._280.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._6_in_D,_K.284_"Dürnitz"/3._Tema_con_variazione/Sonata_No._6Drnitz_3rd_Movement_K._284.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._6_in_D,_K.284_"Dürnitz"/1._Allegro/Sonata_No._6Drnitz_1st_Movement_K._284.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No._6_in_D,_K.284_"Dürnitz"/2._Rondeau_en_Polonaise_(Andante)/Sonata_No._6Rondeau_en_Polonaise_2nd_Movement_K._284.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.16_in_C,_K.545_"Sonata_facile"/2._Andante/Sonata_No._16_2nd_Movement_Mozart.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.16_in_C,_K.545_"Sonata_facile"/3._Rondo_(Allegro)/Sonata_No._16_3rd_Movement_K._545.mxl""","""Wolfgang_Amadeus_Mozart/Piano_Sonata_No.16_in_C,_K.545_"Sonata_facile"/1._Allegro/Sonata_No._16_1st_Movement_K._545.mxl""","""Franz_Liszt/Piano_Sonata_in_B_Minor,_S._178/Sonata_in_B_Minor_S._178.mxl""","""Franz_Liszt/Mephisto_Waltz_No._1,_S.514/musicxml_cleaned.musicxml""","""Alexander_Scriabin/Piano_Sonata_No._5,_Op._53/musicxml_cleaned.musicxml""",]

# remove those in the to_ignore[] list
mask = atepp_metadata.iloc[:,0].isin(to_ignore)
atepp_metadata = atepp_metadata[~mask]

j = 1 # for corpus id

for k, row in atepp_metadata.iterrows():
    performance_id = 'A_' + str(k)
    piece_id = performance_id

    xml_path = str(Path(atepp_path, row['score_path']))
    midi_perfm = str(Path(atepp_path, row['midi_path']))

    feature_file = Path(feature_folder, '{}.pkl'.format(performance_id))
    
    source = 'ATEPP'

    if i % 10 != 0:
        split = 'train'
    else:
        split = 'valid'

    # update metadata file with new dataset information
    metadata = metadata.append({
                    'performance_id': performance_id,
                    'piece_id': piece_id,
                    'source': source,
                    'split': split,
                    'midi_perfm': midi_perfm,
                    'feature_file': str(feature_file),
                    'performance_MIDI_external': '-',
                    'xml_file': xml_path,
                }, ignore_index=True)
    i += 1


# ======== save metadata as new file ==========
metadata.to_csv(str(Path(feature_folder, 'combined_metadata.csv')), index=False)
print('INFO: Metadata saved to {}'.format(Path(feature_folder, 'combined_metadata.csv')))

# after updating the metadata.csv file, will need to generate features for all the new examples for processing/prediction by Pm2S
# ie please run /feature_preperation.py *after* running this file

