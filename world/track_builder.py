import math
import os
import pickle
from world.parameters import SCALE, TRACK_RADIUS, TRACK_DETAIL_STEP, TRACK_TURN_RATE, TOTAL_CHECKPOINTS


class TrackBuilder:
    tracks_directory = os.path.join(os.getcwd(), "tracks")

    @classmethod
    def load_track(cls, env, track_name="track-0.txt"):
        # Form a string for the track file path
        file_path = os.path.join(cls.tracks_directory, track_name)

        # try:
        #     # Try opening the file
        #     with open(file_path, "r") as file:
        #         track = eval(file.read())
        #
        #     # If successful, print success message
        #     print(f"File {file_path} loaded!")
        #
        #     # Return a list of coordinates
        #     return track
        #
        # # If the file with the track name isn't found, we then build
        # # a new track and saved it under the provided track name
        # except FileNotFoundError:
        #     return cls.build_track(env, file_path)

        track_dict = {0: (-0.24655484332525113, -0.10553895239397838, 219.567856160968, -51.6864989969005), 1: (-0.23119196646674656, -0.10553895239397838, 219.9365571430187, -48.20597323405448), 2: (-0.21576923514347524, -0.10553895239397838, 220.30525812506937, -44.72544747120847), 3: (-0.20029367535107845, -0.10553895239397838, 220.67395910712006, -41.244921708362455), 4: (-0.18477246242343526, -0.10553895239397838, 221.04266008917074, -37.76439594551644), 5: (-0.1692129019945634, -0.10553895239397838, 221.41136107122142, -34.28387018267043), 6: (-0.15362241004321753, -0.10553895239397838, 221.7800620532721, -30.80334441982441), 7: (-0.1380084921513811, -0.10553895239397838, 222.1487630353228, -27.32281865697839), 8: (-0.12237872211813983, -0.10553895239397838, 222.51746401737347, -23.842292894132374), 9: (-0.10674072007907487, -0.10553895239397838, 222.88616499942415, -20.361767131286356), 10: (-0.09110213028792735, -0.10553895239397838, 223.25486598147484, -16.88124136844034), 11: (-0.07547059872189266, -0.10553895239397838, 223.62356696352552, -13.400715605594321), 12: (-0.05985375067429555, -0.10553895239397838, 223.9922679455762, -9.920189842748304), 13: (-0.04425916849857359, -0.10553895239397838, 224.36096892762689, -6.439664079902286), 14: (-0.028694369665426933, -0.10553895239397838, 224.72966990967757, -2.9591383170562686), 15: (-0.0131667852907551, -0.10553895239397838, 225.09837089172825, 0.5213874457897489), 16: (0.0023162607143637744, 0.04946104760602162, 225.46707187377893, 4.0019132086357665), 17: (0.017747571728690896, 0.3594610476060216, 224.75643375291557, 7.429010152285853), 18: (0.03304157439091725, 0.6694610476060217, 223.03420369378748, 10.475963322821245), 19: (0.04693571347289101, 0.9794610476060217, 220.464566813903, 12.852297959512285), 20: (0.058230528198600334, 1.2696141176348017, 217.29249394677032, 14.331471284582985), 21: (0.06585936000429013, 1.4070142019369611, 213.84062316600748, 14.909907204037935), 22: (0.06961173119453774, 1.4092612162103406, 210.38618774389764, 15.472824528760519), 23: (0.07341269958091103, 1.4092612162103406, 206.9317523217878, 16.035741853483103), 24: (0.07733833897102595, 1.4092612162103406, 203.47731689967796, 16.598659178205686), 25: (0.08139475823556701, 1.4092612162103406, 200.02288147756812, 17.16157650292827), 26: (0.08558846290918337, 1.4092612162103406, 196.56844605545828, 17.724493827650853), 27: (0.08992638709005159, 1.4092612162103406, 193.11401063334844, 18.287411152373437), 28: (0.09441592837556259, 1.4092612162103406, 189.6595752112386, 18.85032847709602), 29: (0.09906498616427342, 1.4092612162103406, 186.20513978912876, 19.413245801818604), 30: (0.10388200369397248, 1.4092612162103406, 182.75070436701893, 19.976163126541188), 31: (0.1088760142305059, 1.4092612162103406, 179.2962689449091, 20.53908045126377), 32: (0.11405669187255907, 1.4092612162103406, 175.84183352279925, 21.101997775986355), 33: (0.11943440749461237, 1.4092612162103406, 172.3873981006894, 21.66491510070894), 34: (0.12502029041460175, 1.4092612162103406, 168.93296267857957, 22.227832425431522), 35: (0.13082629644529717, 1.4092612162103406, 165.47852725646973, 22.790749750154106), 36: (0.13686528307002416, 1.4092612162103406, 162.0240918343599, 23.35366707487669), 37: (0.14315109257511283, 1.4092612162103406, 158.56965641225005, 23.916584399599273), 38: (0.14969864407442876, 1.4092612162103406, 155.11522099014022, 24.479501724321857), 39: (0.1565240354765825, 1.4092612162103406, 151.66078556803038, 25.04241904904444), 40: (0.16364465657394894, 1.4092612162103406, 148.20635014592054, 25.605336373767024), 41: (0.1710793145753266, 1.4092612162103406, 144.7519147238107, 26.168253698489607), 42: (0.17884837356154196, 1.4092612162103406, 141.29747930170086, 26.73117102321219), 43: (0.1869739095157029, 1.4092612162103406, 137.84304387959102, 27.294088347934775), 44: (0.1954798827665392, 1.4092612162103406, 134.38860845748118, 27.85700567265736), 45: (0.20439232988261644, 1.4092612162103406, 130.93417303537134, 28.419922997379942), 46: (0.2137395772637472, 1.4092612162103406, 127.4797376132615, 28.982840322102525), 47: (0.22355247888772478, 1.4092612162103406, 124.02530219115167, 29.54575764682511), 48: (0.2338646808760273, 1.4092612162103406, 120.57086676904183, 30.108674971547693), 49: (0.24471291572663775, 1.4092612162103406, 117.11643134693199, 30.671592296270276), 50: (0.25613732920349636, 1.4092612162103406, 113.66199592482215, 31.23450962099286), 51: (0.2681818429378575, 1.4092612162103406, 110.20756050271231, 31.797426945715443), 52: (0.2808945557399719, 1.4092612162103406, 106.75312508060247, 32.36034427043803), 53: (0.29432818637293623, 1.4092612162103406, 103.29868965849263, 32.92326159516061), 54: (0.3085405600093883, 1.4092612162103406, 99.8442542363828, 33.486178919883194), 55: (0.32359513964324726, 1.4092612162103406, 96.38981881427296, 34.04909624460578), 56: (0.33956160217893044, 1.4092612162103406, 92.93538339216312, 34.61201356932836), 57: (0.35651645651698494, 1.4092612162103406, 89.48094797005328, 35.174930894050945), 58: (0.3745436973553304, 1.4092612162103406, 86.02651254794344, 35.73784821877353), 59: (0.39373548317073165, 1.4092612162103406, 82.5720771258336, 36.30076554349611), 60: (0.4141928193318501, 1.4092612162103406, 79.11764170372376, 36.863682868218696), 61: (0.43602621674524467, 1.4092612162103406, 75.66320628161392, 37.42660019294128), 62: (0.4593562818761179, 1.4092612162103406, 72.20877085950409, 37.98951751766386), 63: (0.48431417425149087, 1.4092612162103406, 68.75433543739425, 38.55243484238645), 64: (0.5110418413422613, 1.4092612162103406, 65.29990001528441, 39.11535216710903), 65: (0.5396919067450099, 1.4092612162103406, 61.84546459317457, 39.678269491831614), 66: (0.5704270448990942, 1.4092612162103406, 58.39102917106473, 40.2411868165542), 67: (0.6034186241848931, 1.4092612162103406, 54.93659374895489, 40.80410414127678), 68: (0.6388443421446474, 1.4092612162103406, 51.48215832684505, 41.367021465999365), 69: (0.6768845173285303, 1.3796215181669513, 48.027722904735214, 41.92993879072195), 70: (0.717716653322546, 1.327090044199626, 44.61270508820704, 42.69652428027621), 71: (0.7634546271539348, 1.286093871892334, 41.23635055980625, 43.61860350601955), 72: (0.81346520312112, 1.253318528129918, 37.89558919993707, 44.66230534869544), 73: (0.867178641018358, 1.226465209472087, 34.58688564129024, 45.80356814095574), 74: (0.9240307000314251, 1.2039191733663621, 31.306967947595346, 47.025099934478604), 75: (0.9834284941931858, 1.1845262993614527, 28.053072070660335, 48.31434840628896), 76: (1.0447364175709866, 1.1674440872322824, 24.822987260461076, 49.66214368707904), 77: (1.1072789596065915, 1.1520418630646738, 21.615028048994347, 51.061785676148304), 78: (1.170356939437033, 1.1378330365828897, 18.42798717824274, 52.50842678166095), 79: (1.2332728082666327, 1.1244281151445779, 15.261092381316875, 53.99865406137498), 80: (1.2953598788388205, 1.1115010736683086, 12.113976387704723, 55.530209128210835), 81: (1.3560102457737877, 1.098764165772456, 8.98666412884118, 57.10180641618311), 82: (1.4146970162405867, 1.0987697032377166, 5.879579518119153, 58.713024968029615), 83: (1.47098809784387, 1.1092982311901811, 2.7522498541166276, 60.28458762113952), 84: (1.5251737196537762, 1.1162840949366282, -0.38807095395105984, 61.83002775987577), 85: (1.577072660031686, 1.1210666363780897, -3.537052111958186, 63.35774424748598), 86: (1.6265651118707793, 1.124441793574948, -6.692011303484829, 64.87307671919031), 87: (1.6735881994835795, 1.1268950602914702, -9.851233205632255, 66.37950194899496), 88: (1.7181287090818549, 1.128730102576374, -13.01358942048308, 67.87933632998319), 89: (1.7602144528987202, 1.1301415055497281, -16.17831874163188, 69.37415682885958), 90: (1.7999054353392299, 1.131256905400765, -19.344896093082006, 70.86505852826271), 91: (1.8372855932541734, 1.1321619556377769, -22.512952182005822, 72.35281543630674), 92: (1.8724555382113184, 1.13291545207102, -25.682223026268574, 73.83798287323299), 93: (1.905526475467635, 1.1335586946823941, -28.852517556020686, 75.32096385578934), 94: (1.9366153070082326, 1.1341214064089353, -32.02369642307597, 76.80205284472349), 95: (1.9658408272126036, 1.1346255566255157, -35.19565792634165, 78.28146496506066), 96: (1.9933208705662566, 1.1350878886666491, -38.36832856379228, 79.7593557006222), 97: (2.019170254813341, 1.1355216355777413, -41.54165667009432, 81.23583419005357), 98: (2.0434993671069956, 1.1359377242245916, -44.71560817467763, 82.71097209769249), 99: (2.0664132557618617, 1.1363456590672962, -47.89016387391637, 84.18480930458843), 100: (2.0880111099518412, 1.1672817454042268, -51.06531784230702, 85.65735719074125), 101: (2.108386030311836, 1.2311268603697751, -54.32493045690722, 86.93208288779896), 102: (2.1293325959574543, 1.4192389409161783, -57.6617538896894, 87.98830700153417), 103: (2.1509123456543815, 1.7292389409161784, -61.1617331494239, 87.97625787590994), 104: (2.1782984391456313, 2.0392389409161784, -64.49120520253956, 86.89708418853527), 105: (2.2092584402061055, 2.3492389409161785, -67.33276195516174, 84.85366665371102), 106: (2.2415712297011576, 2.6159092303018463, -69.41550967456118, 82.04081010967319), 107: (2.2730274206951937, 2.7963303288222523, -70.82351290380142, 78.8365119171747), 108: (2.302704219318079, 2.9070046975545756, -71.77901761807392, 75.46946413892697), 109: (2.3311370222802665, 2.9749506984883975, -72.44917896346433, 72.0342226960085), 110: (2.359066481866057, 3.017572661248469, -72.93972685439883, 68.56876994540004), 111: (2.3870729724874975, 3.045084321584624, -73.31497585365885, 65.0889440605957), 112: (2.4155596157142916, 3.063392041410249, -73.6141968327812, 61.601757973554584), 113: (2.4448007907657088, 3.0759530917055034, -73.86180742794163, 58.11052767562746), 114: (2.474985444741798, 3.0848341747930172, -74.07333776484296, 54.61692567817978), 115: (2.5062460059361618, 3.0913002683084096, -74.25890074910696, 51.12184822855566), 116: (2.538675959193051, 3.0961444473808717, -74.42523474280664, 47.62580289095778), 117: (2.572339967699912, 3.099875820453321, -74.57692798592365, 44.12909169899573), 118: (2.6072793449061473, 3.102828839499591, -74.71716749116479, 40.631902415990055), 119: (2.6435146779869765, 3.1052282542622307, -74.84820649922295, 37.13435630787838), 120: (2.681046739386232, 3.107228542290896, -74.9716620389221, 33.63653430987482), 121: (2.719856424566561, 3.108938354536655, -75.08870783647136, 30.138491960034546), 122: (2.759904225460685, 3.110435992397245, -75.20020143987782, 26.640268242717507), 123: (2.801129619159633, 3.111779419272791, -75.30676912672374, 23.14189100060649), 124: (2.8434506736131224, 3.113012888040706, -75.40886312347614, 19.64338034379449), 125: (2.886764116825037, 3.114171447267802, -75.50680021666903, 16.144750851296475), 126: (2.9309460644103678, 3.1152841101847732, -75.60078748299719, 12.646013022626233), 127: (2.975853541461866, 3.1163761871136995, -75.69093874952702, 9.147174251098876), 128: (3.021326863942755, 3.030080022991815, -75.7772840160206, 5.648239485351714), 129: (3.067192863601006, 2.873023271287875, -76.46697504812235, 2.2168656671502585), 130: (3.112609620108855, 2.7461317468530755, -77.63017351976198, -1.084190120862909), 131: (3.155557837343855, 2.643351775374952, -79.15928030618129, -4.23249640134555), 132: (3.1950098896865873, 2.559216076552066, -80.97178592536281, -7.226627891516854), 133: (3.230605530010351, 2.489263865010262, -83.00656855447372, -10.074374307943101), 134: (3.2623703339506864, 2.430022243922823, -85.21889183910154, -12.786499984288785), 135: (3.2905246753356976, 2.3788363566644497, -87.57603008509574, -15.373757081925476), 136: (3.315369545310503, 2.3336795102010224, -90.05393135789507, -17.845599569273617), 137: (3.337223834596526, 2.2929889845691784, -92.63481028515048, -20.20971969372774), 138: (3.356392477159896, 2.255536501955114, -95.30544983924699, -22.471949649345696), 139: (3.3731519213273615, 2.3160551443891997, -98.05601059910262, -24.63630070828747), 140: (3.387744832061969, 2.4492838711128737, -100.43396545978868, -27.20443829816453), 141: (3.4061141770668346, 2.5422769958337987, -102.51755093175316, -30.016674344626235), 142: (3.42642675054852, 2.6080206609258134, -104.37960704623777, -32.9802443246521), 143: (3.4476306194160977, 2.6554959780236658, -106.07649815071298, -36.04138146296408), 144: (3.4691230323407902, 2.690638563740955, -107.64919655399845, -39.168140125269716), 145: (3.4905521659536443, 2.7173382422372683, -109.12692260823138, -42.340887469235376), 146: (3.5117087373414506, 2.7381636432576064, -110.53049171378282, -45.547130405436796), 147: (3.532467129252383, 2.7548362370027024, -111.87482691476481, -48.77865680443166), 148: (3.552752640867856, 2.7685309475908197, -113.17072281709684, -52.02991099478945), 149: (3.572522637451994, 2.780066046562877, -114.42605523101328, -55.29704131520379), 150: (3.5917553012600814, 2.790024073075024, -115.64660586432723, -58.577324252926836), 151: (3.610442685221227, 2.7988299187036114, -116.83662528396057, -61.868806256659425), 152: (3.6285862915562106, 2.8068021602264763, -117.999218288177, -65.17007533181022), 153: (3.646194185773921, 2.8141875487099792, -119.13660768839962, -68.48011239387785), 154: (3.6632790775486503, 2.8211848263597337, -120.25031333395208, -71.79819312452088), 155: (3.679857032375979, 2.8279617867413167, -121.34127037381425, -75.12382279981173), 156: (3.695946610593096, 2.8346681328758807, -122.40990209585433, -78.4566933707535), 157: (3.711568308858156, 2.9930215867168553, -123.45615669541124, -81.79665614193914), 158: (3.7267442275495912, 3.3030215867168558, -123.43365558446685, -85.29658381262125), 159: (3.7462743132157716, 3.6130215867168554, -122.34454385828582, -88.62281826736057), 160: (3.7684883183215674, 3.9230215867168554, -120.29264965283348, -91.45826006179598), 161: (3.7916519657524463, 4.2330215867168555, -117.47358591160202, -93.53259841965232), 162: (3.814012897494412, 4.543021586716854, -114.1561020442097, -94.64808073219329), 163: (3.833833885053869, 4.764036746631462, -110.65646327649719, -94.69836488002998), 164: (3.8494368012920495, 4.85600927891006, -107.18066332457053, -94.28749422619039), 165: (3.863081904174051, 4.892555252241003, -103.73086687081975, -93.69681290120329), 166: (3.8762105627703156, 4.9076749837988, -100.29435200268945, -93.0332124482805), 167: (3.889449723808691, 4.914239473255712, -96.86399188307225, -92.3384970750953), 168: (3.9030765857092438, 4.917227942084916, -93.43648533293461, -91.62983717869673), 169: (3.9172289611191338, 4.9186517373555745, -90.01033873161002, -90.91463128789884), 170: (3.931988938728359, 4.919360254502782, -86.58486404838817, -90.19621414263702), 171: (3.947416350624047, 4.919600086342463, -83.1597343578262, -89.4761540042938), 172: (3.9635626062815392, 4.919600086342463, -79.73460466726422, -88.75609386595059), 173: (3.980482689226163, 4.919600086342463, -76.30947497670225, -88.03603372760737), 174: (3.9982231762985645, 4.919600086342463, -72.88434528614027, -87.31597358926416), 175: (4.016833199688547, 4.919600086342463, -69.4592155955783, -86.59591345092095), 176: (4.036364410440911, 4.919600086342463, -66.03408590501633, -85.87585331257773), 177: (4.056870880778225, 4.919600086342463, -62.60895621445435, -85.15579317423452), 178: (4.078408927949985, 4.919600086342463, -59.18382652389238, -84.4357330358913), 179: (4.101036839084438, 4.919600086342463, -55.758696833330404, -83.71567289754809), 180: (4.124814473116574, 4.919600086342463, -52.33356714276843, -82.99561275920487), 181: (4.149802712504646, 4.919600086342463, -48.908437452206456, -82.27555262086166), 182: (4.176062734466965, 4.919600086342463, -45.48330776164448, -81.55549248251845), 183: (4.203655069375813, 4.919600086342463, -42.05817807108251, -80.83543234417523), 184: (4.232638413446588, 4.919600086342463, -38.633048380520535, -80.11537220583202), 185: (4.263068164901858, 4.919600086342463, -35.20791868995856, -79.3953120674888), 186: (4.294994658551751, 4.919600086342463, -31.78278899939659, -78.67525192914559), 187: (4.32846108458236, 4.919600086342463, -28.35765930883462, -77.95519179080237), 188: (4.36350109471044, 4.919600086342463, -24.93252961827265, -77.23513165245916), 189: (4.400136123989431, 4.919600086342463, -21.50739992771068, -76.51507151411595), 190: (4.438372490111786, 4.919600086342463, -18.08227023714871, -75.79501137577273), 191: (4.478198373618121, 4.919600086342463, -14.657140546586739, -75.07495123742952), 192: (4.5195808298423215, 4.919600086342463, -11.232010856024768, -74.3548910990863), 193: (4.562463032231893, 4.919600086342463, -7.806881165462797, -73.63483096074309), 194: (4.606761989797768, 4.9276425980689975, -4.381751474900826, -72.91477082239987), 195: (4.65236700941799, 4.942273153221298, -0.9686465480079578, -72.13971291616342), 196: (4.698962418213135, 4.954394866777038, 2.4339501750523453, -71.3197522218498), 197: (4.74650303492035, 4.964692034086545, 5.827263916224606, -70.46218481693546), 198: (4.794901795058712, 4.9736556088339885, 9.212253747600279, -69.57232873785115), 199: (4.844035764602266, 4.9816468674954795, 12.589649311621034, -68.65406994854777), 200: (4.893752014750631, 4.988939524641783, 15.959985285122096, -67.71022916445088), 201: (4.943873776900643, 4.995748260864201, 19.323628998285947, -66.74280941537951), 202: (4.9942070492664365, 5.0022486407142175, 22.68080015790562, -65.75316210192996), 203: (5.044547613014992, 5.008591601260815, 26.031582776568076, -64.74209541645884), 204: (5.094688251547074, 5.068610374195416, 29.375929549688166, -63.70994006908073), 205: (5.1444258630834545, 5.177743018057584, 32.581525694329464, -62.304894391011565), 206: (5.194217250905308, 5.27853657092215, 35.623027125985594, -60.573054605917775), 207: (5.2440126957627005, 5.372671372294519, 38.4823854214418, -58.55462201150056), 208: (5.293812504860334, 5.461697253277494, 41.14601218353751, -56.284139139456045), 209: (5.343645492332951, 5.547034578682572, 43.60267620166429, -53.7911886983038), 210: (5.393554824051426, 5.629994388675238, 45.8419663016066, -51.10128538251337), 211: (5.443588751516204, 5.516037793944154, 47.853158501182335, -48.236829773437584), 212: (5.493794332041359, 5.206037793944155, 50.642311270588024, -46.12244408749555), 213: (5.544463067252032, 4.896037793944155, 53.94352669870178, -44.959698759754474), 214: (5.588373080119499, 4.586037793944156, 57.44209047653083, -44.85944162802532), 215: (5.620168192189423, 4.3505550772326735, 60.80447451244264, -45.83123049143415), 216: (5.6372900062685085, 4.2614497419002415, 63.967643250193916, -47.32935049213936), 217: (5.646182716268309, 4.251768500613451, 67.10450748496638, -48.88179462079758), 218: (5.653624162663183, 4.250557743901627, 70.23807778625884, -50.44087677606005), 219: (5.66038733553261, 4.250405609855642, 73.3711735635545, -52.000912304595246), 220: (5.666619195773136, 4.250405609855642, 76.50426934085017, -53.56094783313044), 221: (5.6723895225388565, 4.250405609855642, 79.63736511814584, -55.12098336166564), 222: (5.677747411939052, 4.250405609855642, 82.7704608954415, -56.68101889020083), 223: (5.682735279782875, 4.250405609855642, 85.90355667273717, -58.24105441873603), 224: (5.687389953350199, 4.250405609855642, 89.03665245003283, -59.80108994727122), 225: (5.691743557119236, 4.250405609855642, 92.1697482273285, -61.36112547580642), 226: (5.695824236163651, 4.250405609855642, 95.30284400462416, -62.92116100434161), 227: (5.699656750707682, 4.250405609855642, 98.43593978191983, -64.48119653287681), 228: (5.7032629677015105, 4.250405609855642, 101.5690355592155, -66.041232061412), 229: (5.706662269541914, 4.250405609855642, 104.70213133651116, -67.6012675899472), 230: (5.709871895711908, 4.250405609855642, 107.83522711380682, -69.1613031184824), 231: (5.7129072297873, 4.250405609855642, 110.96832289110249, -70.72133864701759), 232: (5.715782041697441, 4.250405609855642, 114.10141866839815, -72.28137417555278), 233: (5.718508693142059, 4.250405609855642, 117.23451444569382, -73.84140970408798), 234: (5.721098312516462, 4.250405609855642, 120.36761022298948, -75.40144523262317), 235: (5.7235609444802105, 4.250405609855642, 123.50070600028515, -76.96148076115837), 236: (5.725905678342534, 4.250405609855642, 126.63380177758081, -78.52151628969357), 237: (5.728140758673286, 4.250405609855642, 129.76689755487646, -80.08155181822876), 238: (5.7302736809372865, 4.250405609855642, 132.89999333217213, -81.64158734676396), 239: (5.732311274459109, 4.250405609855642, 136.0330891094678, -83.20162287529915), 240: (5.734259774629018, 4.250405609855642, 139.16618488676346, -84.76165840383435), 241: (5.736124885939283, 4.250405609855642, 142.29928066405913, -86.32169393236954), 242: (5.737911837177985, 4.250405609855642, 145.4323764413548, -87.88172946090474), 243: (5.7396254298928815, 4.250405609855642, 148.56547221865046, -89.44176498943993), 244: (5.741270081061544, 4.250405609855642, 151.69856799594612, -91.00180051797513), 245: (5.742849860758313, 4.250405609855642, 154.8316637732418, -92.56183604651032), 246: (5.744368525487976, 4.250405609855642, 157.96475955053745, -94.12187157504552), 247: (5.745829547755695, 4.250405609855642, 161.09785532783312, -95.68190710358071), 248: (5.747236142358875, 4.250405609855642, 164.23095110512878, -97.24194263211591), 249: (5.748591289816505, 4.250405609855642, 167.36404688242445, -98.8019781606511), 250: (5.749897757292445, 4.250405609855642, 170.4971426597201, -100.3620136891863), 251: (5.751158117319385, 4.250405609855642, 173.63023843701578, -101.9220492177215), 252: (5.752374764588119, 4.250405609855642, 176.76333421431144, -103.48208474625669), 253: (5.753549931031033, 4.250405609855642, 179.8964299916071, -105.04212027479188), 254: (5.754685699398336, 4.250405609855642, 183.02952576890277, -106.60215580332708), 255: (5.755784015499606, 4.250405609855642, 186.16262154619844, -108.16219133186227), 256: (5.756846699261073, 4.250405609855642, 189.2957173234941, -109.72222686039747), 257: (5.757875454729973, 4.250405609855642, 192.42881310078977, -111.28226238893266), 258: (5.758871879140984, 4.250405609855642, 195.56190887808543, -112.84229791746786), 259: (-0.5233478360339685, -1.877779697323944, 198.6950046553811, -114.40233344600306), 260: (-0.5224116688863463, -1.567779697323944, 202.1546592529052, -114.932229724373), 261: (-0.5169629495044878, -1.257779697323944, 205.61105390247283, -114.3814702247362), 262: (-0.5076673445809528, -0.9477796973239438, 208.73468061606198, -112.80256042549111), 263: (-0.4954515779739319, -0.6377796973239438, 211.22775512477983, -110.34602231384267), 264: (-0.48140899608462107, -0.36569216124821946, 212.85260550494735, -107.24604469423483), 265: (-0.4667238353806127, -0.20476223347072242, 213.71378653486448, -103.85364624568682), 266: (-0.4523424133127518, -0.14442080405948138, 214.27457834678262, -100.39886513197128), 267: (-0.43817453327461386, -0.12141052074696004, 214.72108443079648, -96.92746306408306), 268: (-0.42402670912266327, -0.11219547425962009, 215.12234762972417, -93.45054088416785), 269: (-0.4098109626348583, -0.1083123096803307, 215.50480324287028, -89.97149967947179), 270: (-0.39549403117499615, -0.10659376929300338, 215.8790516899316, -86.4915660450655), 271: (-0.3810650954501611, -0.10579677250125077, 216.24954732191054, -83.01123086197113), 272: (-0.3665223956596275, -0.10553895240182304, 216.61824830398854, -79.530705099128), 273: (-0.3518679946579093, -0.10553895240182304, 216.98694928606653, -76.05017933628487), 274: (-0.33710480764442785, -0.10553895240182304, 217.35565026814453, -72.56965357344174), 275: (-0.3222381235065237, -0.10553895240182304, 217.72435125022253, -69.08912781059861), 276: (-0.30727348840533697, -0.10553895240182304, 218.09305223230052, -65.60860204775548), 277: (-0.29221669716900056, -0.10553895240182304, 218.46175321437852, -62.12807628491235), 278: (-0.2770737832213568, -0.10553895240182304, 218.83045419645651, -58.647550522069224)}
        track = []
        for value in track_dict.values():
            track.append(value)

        return track

    @classmethod
    def build_track(cls, env, file_path):
        # Create checkpoint nodes
        checkpoint_nodes = [
            cls._create_checkpoint_node(env, i, TOTAL_CHECKPOINTS, TRACK_RADIUS) for i in range(TOTAL_CHECKPOINTS)]

        # Generate track coordinates which connect the checkpoint nodes
        track_edges = cls._create_track_edges(checkpoint_nodes, TRACK_RADIUS, TRACK_TURN_RATE, TRACK_DETAIL_STEP)

        # Save track under the provided track name
        success = cls.save_track(track_edges, file_path)

        return track_edges if success is True else None

    @classmethod
    def save_track(cls, track, file_path):
        try:
            with open(file_path, "w") as file:
                file.write(str(track))

            return True
        except IOError:
            print(f"File {file_path} has not been saved")

            return False

    @staticmethod
    def _create_checkpoint_node(env, checkpoint_index, total_checkpoints, track_radius):
        if checkpoint_index == 0:
            alpha = 0
            radius = 1.5 * track_radius
        elif checkpoint_index == total_checkpoints - 1:
            # Set alpha as the total angle in a circle (2*pi) divided by the number of checkpoints
            # and multiplied with the index of the current checkpoint, which is the last index
            alpha = 2 * math.pi * checkpoint_index / total_checkpoints
            radius = 1.5 * track_radius
            env.start_alpha = 2 * math.pi * (-0.5) / total_checkpoints
        else:
            # Similar to alpha above, but this time we add it by a random angle
            # between 0 and the elementary angle of each checkpoint (1/total_checkpoints)
            alpha = 2 * math.pi * checkpoint_index / total_checkpoints + \
                    env.np_random.uniform(0, 2 * math.pi * 1 / total_checkpoints)
            # Randomly pick the radius between a third of the track radius and full radius
            radius = env.np_random.uniform(track_radius / 3, track_radius)

        # Return result in cartesian coordinates
        return alpha, radius * math.cos(alpha), radius * math.sin(alpha)

    @staticmethod
    def _create_track_edges(checkpoint_nodes, track_radius, track_turn_rate, track_detail_step):
        x, y, beta = 1.5 * track_radius, 0, 0
        dest_i = 0
        laps = 0
        track_edges = []
        no_freeze = 2500
        # Whether we are journeying from
        visited_other_side = False

        # No idea
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False

            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            # Find destination from checkpoints (no idea)
            while True:
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoint_nodes[dest_i % len(checkpoint_nodes)]
                    if alpha <= dest_alpha:
                        failed = False
                        break

                    dest_i += 1
                    if dest_i % len(checkpoint_nodes) == 0:
                        break
                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            # Declare variables which I don't understand
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x
            dest_dy = dest_y - y

            # Destination vector projected on radius
            projection = r1x * dest_dx + r1y * dest_dy

            # Normalise beta-alpha: if they are larger than 1.5pi, reduce beta by
            # a full revolution of 2pi until they are smaller than 1.5pi.
            # Similarly, if they are smaller than -1.5pi, add beta by a full revolution
            # until it is larger than 1.5pi
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi

            # No idea
            prev_beta = beta
            projection *= SCALE

            # Normalise beta again by checking the projection value. If it is larger
            # than 0.3, reduce it once by min() function below. Conversely if projection
            # is smaller than -0.3, add it once by min() function below
            if projection > 0.3:
                beta -= min(track_turn_rate, abs(0.001 * projection))
            if projection < -0.3:
                beta += min(track_turn_rate, abs(0.001 * projection))

            # x and y go further into "one detail-step" (similar to one time-step)
            x += p1x * track_detail_step
            y += p1y * track_detail_step

            # Append edge to list of tracks
            track_edges.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))

            # No idea
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        return track_edges
