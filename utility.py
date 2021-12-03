class Defaults(object):
    """docstring for defaults"""
    time_bits = 23
    time_mask = 0x3fffff
    data_bits = 40

    param_set = {
        'i_thres': 40000,
        't_ltp': 2 * 10 ** 3,
        't_refrac': 10 * 10 ** 3,
        't_inhibit': 15 * 10 ** 2,
        't_leak': 5 * 10 ** 3,
        'w_min': 1,
        'w_max': 1000,
        'a_dec': 50,
        'a_inc': 100,
        'randmut': 0.001
    }

    UP = 0
    DOWN = 1
    RIGHT = 2
    UPLEFT = 3
    UPRIGHT = 4
    DOWNLEFT = 5
    DOWNRIGHT = 6

    resources_folder = "resources"

    files = [f"{resources_folder}\\trace_up.bin",
             f"{resources_folder}\\trace_down.bin",
             f"{resources_folder}\\trace_right.bin",
             f"{resources_folder}\\trace_upleft.bin",
             f"{resources_folder}\\trace_upright.bin",
             f"{resources_folder}\\trace_downleft.bin",
             f"{resources_folder}\\trace_downright.bin"]

    traces = ["trace_up",
             "trace_down",
             "trace_right",
             "trace_upleft",
             "trace_upright",
             "trace_downleft",
             "trace_downright"]

    pixels = ['00000', '00008', '00010', '00018', '00020', '00028', '00030', '00038', '00040', '00048', '00050', '00058', '00060', '00068', '00070', '00078', '00080', '00088', '00090', '00098', '000a0', '000a8', '000b0', '000b8', '000c0', '000c8', '000d0', '000d8', '000e0', '000e8', '000f0', '000f8', '00100', '00108', '00110', '00118', '00120', '00128', '00130', '00138', '00140', '00148', '00150', '00158', '00160', '00168', '00170', '00178', '00180', '00188', '00190', '00198', '001a0', '001a8', '001b0', '001b8', '01000', '01008', '01010', '01018', '01020', '01028', '01030', '01038', '01040', '01048', '01050', '01058', '01060', '01068', '01070', '01078', '01080', '01088', '01090', '01098', '010a0', '010a8', '010b0', '010b8', '010c0', '010c8', '010d0', '010d8', '010e0', '010e8', '010f0', '010f8', '01100', '01108', '01110', '01118', '01120', '01128', '01130', '01138', '01140', '01148', '01150', '01158', '01160', '01168', '01170', '01178', '01180', '01188', '01190', '01198', '011a0', '011a8', '011b0', '011b8', '02000', '02008', '02010', '02018', '02020', '02028', '02030', '02038', '02040', '02048', '02050', '02058', '02060', '02068', '02070', '02078', '02080', '02088', '02090', '02098', '020a0', '020a8', '020b0', '020b8', '020c0', '020c8', '020d0', '020d8', '020e0', '020e8', '020f0', '020f8', '02100', '02108', '02110', '02118', '02120', '02128', '02130', '02138', '02140', '02148', '02150', '02158', '02160', '02168', '02170', '02178', '02180', '02188', '02190', '02198', '021a0', '021a8', '021b0', '021b8', '03000', '03008', '03010', '03018', '03020', '03028', '03030', '03038', '03040', '03048', '03050', '03058', '03060', '03068', '03070', '03078', '03080', '03088', '03090', '03098', '030a0', '030a8', '030b0', '030b8', '030c0', '030c8', '030d0', '030d8', '030e0', '030e8', '030f0', '030f8', '03100', '03108', '03110', '03118', '03120', '03128', '03130', '03138', '03140', '03148', '03150', '03158', '03160', '03168', '03170', '03178', '03180', '03188', '03190', '03198', '031a0', '031a8', '031b0', '031b8', '04000', '04008', '04010', '04018', '04020', '04028', '04030', '04038', '04040', '04048', '04050', '04058', '04060', '04068', '04070', '04078', '04080', '04088', '04090', '04098', '040a0', '040a8', '040b0', '040b8', '040c0', '040c8', '040d0', '040d8', '040e0', '040e8', '040f0', '040f8', '04100', '04108', '04110', '04118', '04120', '04128', '04130', '04138', '04140', '04148', '04150', '04158', '04160', '04168', '04170', '04178', '04180', '04188', '04190', '04198', '041a0', '041a8', '041b0', '041b8', '05000', '05008', '05010', '05018', '05020', '05028', '05030', '05038', '05040', '05048', '05050', '05058', '05060', '05068', '05070', '05078', '05080', '05088', '05090', '05098', '050a0', '050a8', '050b0', '050b8', '050c0', '050c8', '050d0', '050d8', '050e0', '050e8', '050f0', '050f8', '05100', '05108', '05110', '05118', '05120', '05128', '05130', '05138', '05140', '05148', '05150', '05158', '05160', '05168', '05170', '05178', '05180', '05188', '05190', '05198', '051a0', '051a8', '051b0', '051b8', '06000', '06008', '06010', '06018', '06020', '06028', '06030', '06038', '06040', '06048', '06050', '06058', '06060', '06068', '06070', '06078', '06080', '06088', '06090', '06098', '060a0', '060a8', '060b0', '060b8', '060c0', '060c8', '060d0', '060d8', '060e0', '060e8', '060f0', '060f8', '06100', '06108', '06110', '06118', '06120', '06128', '06130', '06138', '06140', '06148', '06150', '06158', '06160', '06168', '06170', '06178', '06180', '06188', '06190', '06198', '061a0', '061a8', '061b0', '061b8', '07000', '07008', '07010', '07018', '07020', '07028', '07030', '07038', '07040', '07048', '07050', '07058', '07060', '07068', '07070', '07078', '07080', '07088', '07090', '07098', '070a0', '070a8', '070b0', '070b8', '070c0', '070c8', '070d0', '070d8', '070e0', '070e8', '070f0', '070f8', '07100', '07108', '07110', '07118', '07120', '07128', '07130', '07138', '07140', '07148', '07150', '07158', '07160', '07168', '07170', '07178', '07180', '07188', '07190', '07198', '071a0', '071a8', '071b0', '071b8', '08000', '08008', '08010', '08018', '08020', '08028', '08030', '08038', '08040', '08048', '08050', '08058', '08060', '08068', '08070', '08078', '08080', '08088', '08090', '08098', '080a0', '080a8', '080b0', '080b8', '080c0', '080c8', '080d0', '080d8', '080e0', '080e8', '080f0', '080f8', '08100', '08108', '08110', '08118', '08120', '08128', '08130', '08138', '08140', '08148', '08150', '08158', '08160', '08168', '08170', '08178', '08180', '08188', '08190', '08198', '081a0', '081a8', '081b0', '081b8', '09000', '09008', '09010', '09018', '09020', '09028', '09030', '09038', '09040', '09048', '09050', '09058', '09060', '09068', '09070', '09078', '09080', '09088', '09090', '09098', '090a0', '090a8', '090b0', '090b8', '090c0', '090c8', '090d0', '090d8', '090e0', '090e8', '090f0', '090f8', '09100', '09108', '09110', '09118', '09120', '09128', '09130', '09138', '09140', '09148', '09150', '09158', '09160', '09168', '09170', '09178', '09180', '09188', '09190', '09198', '091a0', '091a8', '091b0', '091b8', '0a000', '0a008', '0a010', '0a018', '0a020', '0a028', '0a030', '0a038', '0a040', '0a048', '0a050', '0a058', '0a060', '0a068', '0a070', '0a078', '0a080', '0a088', '0a090', '0a098', '0a0a0', '0a0a8', '0a0b0', '0a0b8', '0a0c0', '0a0c8', '0a0d0', '0a0d8', '0a0e0', '0a0e8', '0a0f0', '0a0f8', '0a100', '0a108', '0a110', '0a118', '0a120', '0a128', '0a130', '0a138', '0a140', '0a148', '0a150', '0a158', '0a160', '0a168', '0a170', '0a178', '0a180', '0a188', '0a190', '0a198', '0a1a0', '0a1a8', '0a1b0', '0a1b8', '0b000', '0b008', '0b010', '0b018', '0b020', '0b028', '0b030', '0b038', '0b040', '0b048', '0b050', '0b058', '0b060', '0b068', '0b070', '0b078', '0b080', '0b088', '0b090', '0b098', '0b0a0', '0b0a8', '0b0b0', '0b0b8', '0b0c0', '0b0c8', '0b0d0', '0b0d8', '0b0e0', '0b0e8', '0b0f0', '0b0f8', '0b100', '0b108', '0b110', '0b118', '0b120', '0b128', '0b130', '0b138', '0b140', '0b148', '0b150', '0b158', '0b160', '0b168', '0b170', '0b178', '0b180', '0b188', '0b190', '0b198', '0b1a0', '0b1a8', '0b1b0', '0b1b8', '0c000', '0c008', '0c010', '0c018', '0c020', '0c028', '0c030', '0c038', '0c040', '0c048', '0c050', '0c058', '0c060', '0c068', '0c070', '0c078', '0c080', '0c088', '0c090', '0c098', '0c0a0', '0c0a8', '0c0b0', '0c0b8', '0c0c0', '0c0c8', '0c0d0', '0c0d8', '0c0e0', '0c0e8', '0c0f0', '0c0f8', '0c100', '0c108', '0c110', '0c118', '0c120', '0c128', '0c130', '0c138', '0c140', '0c148', '0c150', '0c158', '0c160', '0c168', '0c170', '0c178', '0c180', '0c188', '0c190', '0c198', '0c1a0', '0c1a8', '0c1b0', '0c1b8', '0d000', '0d008', '0d010', '0d018', '0d020', '0d028', '0d030', '0d038', '0d040', '0d048', '0d050', '0d058', '0d060', '0d068', '0d070', '0d078', '0d080', '0d088', '0d090', '0d098', '0d0a0', '0d0a8', '0d0b0', '0d0b8', '0d0c0', '0d0c8', '0d0d0', '0d0d8', '0d0e0', '0d0e8', '0d0f0', '0d0f8', '0d100', '0d108', '0d110', '0d118', '0d120', '0d128', '0d130', '0d138', '0d140', '0d148', '0d150', '0d158', '0d160', '0d168', '0d170', '0d178', '0d180', '0d188', '0d190', '0d198', '0d1a0', '0d1a8', '0d1b0', '0d1b8', '0e000', '0e008', '0e010', '0e018', '0e020', '0e028', '0e030', '0e038', '0e040', '0e048', '0e050', '0e058', '0e060', '0e068', '0e070', '0e078', '0e080', '0e088', '0e090', '0e098', '0e0a0', '0e0a8', '0e0b0', '0e0b8', '0e0c0', '0e0c8', '0e0d0', '0e0d8', '0e0e0', '0e0e8', '0e0f0', '0e0f8', '0e100', '0e108', '0e110', '0e118', '0e120', '0e128', '0e130', '0e138', '0e140', '0e148', '0e150', '0e158', '0e160', '0e168', '0e170', '0e178', '0e180', '0e188', '0e190', '0e198', '0e1a0', '0e1a8', '0e1b0', '0e1b8', '0f000', '0f008', '0f010', '0f018', '0f020', '0f028', '0f030', '0f038', '0f040', '0f048', '0f050', '0f058', '0f060', '0f068', '0f070', '0f078', '0f080', '0f088', '0f090', '0f098', '0f0a0', '0f0a8', '0f0b0', '0f0b8', '0f0c0', '0f0c8', '0f0d0', '0f0d8', '0f0e0', '0f0e8', '0f0f0', '0f0f8', '0f100', '0f108', '0f110', '0f118', '0f120', '0f128', '0f130', '0f138', '0f140', '0f148', '0f150', '0f158', '0f160', '0f168', '0f170', '0f178', '0f180', '0f188', '0f190', '0f198', '0f1a0', '0f1a8', '0f1b0', '0f1b8', '10000', '10008', '10010', '10018', '10020', '10028', '10030', '10038', '10040', '10048', '10050', '10058', '10060', '10068', '10070', '10078', '10080', '10088', '10090', '10098', '100a0', '100a8', '100b0', '100b8', '100c0', '100c8', '100d0', '100d8', '100e0', '100e8', '100f0', '100f8', '10100', '10108', '10110', '10118', '10120', '10128', '10130', '10138', '10140', '10148', '10150', '10158', '10160', '10168', '10170', '10178', '10180', '10188', '10190', '10198', '101a0', '101a8', '101b0', '101b8', '11000', '11008', '11010', '11018', '11020', '11028', '11030', '11038', '11040', '11048', '11050', '11058', '11060', '11068', '11070', '11078', '11080', '11088', '11090', '11098', '110a0', '110a8', '110b0', '110b8', '110c0', '110c8', '110d0', '110d8', '110e0', '110e8', '110f0', '110f8', '11100', '11108', '11110', '11118', '11120', '11128', '11130', '11138', '11140', '11148', '11150', '11158', '11160', '11168', '11170', '11178', '11180', '11188', '11190', '11198', '111a0', '111a8', '111b0', '111b8', '12000', '12008', '12010', '12018', '12020', '12028', '12030', '12038', '12040', '12048', '12050', '12058', '12060', '12068', '12070', '12078', '12080', '12088', '12090', '12098', '120a0', '120a8', '120b0', '120b8', '120c0', '120c8', '120d0', '120d8', '120e0', '120e8', '120f0', '120f8', '12100', '12108', '12110', '12118', '12120', '12128', '12130', '12138', '12140', '12148', '12150', '12158', '12160', '12168', '12170', '12178', '12180', '12188', '12190', '12198', '121a0', '121a8', '121b0', '121b8', '13000', '13008', '13010', '13018', '13020', '13028', '13030', '13038', '13040', '13048', '13050', '13058', '13060', '13068', '13070', '13078', '13080', '13088', '13090', '13098', '130a0', '130a8', '130b0', '130b8', '130c0', '130c8', '130d0', '130d8', '130e0', '130e8', '130f0', '130f8', '13100', '13108', '13110', '13118', '13120', '13128', '13130', '13138', '13140', '13148', '13150', '13158', '13160', '13168', '13170', '13178', '13180', '13188', '13190', '13198', '131a0', '131a8', '131b0', '131b8', '14000', '14008', '14010', '14018', '14020', '14028', '14030', '14038', '14040', '14048', '14050', '14058', '14060', '14068', '14070', '14078', '14080', '14088', '14090', '14098', '140a0', '140a8', '140b0', '140b8', '140c0', '140c8', '140d0', '140d8', '140e0', '140e8', '140f0', '140f8', '14100', '14108', '14110', '14118', '14120', '14128', '14130', '14138', '14140', '14148', '14150', '14158', '14160', '14168', '14170', '14178', '14180', '14188', '14190', '14198', '141a0', '141a8', '141b0', '141b8', '15000', '15008', '15010', '15018', '15020', '15028', '15030', '15038', '15040', '15048', '15050', '15058', '15060', '15068', '15070', '15078', '15080', '15088', '15090', '15098', '150a0', '150a8', '150b0', '150b8', '150c0', '150c8', '150d0', '150d8', '150e0', '150e8', '150f0', '150f8', '15100', '15108', '15110', '15118', '15120', '15128', '15130', '15138', '15140', '15148', '15150', '15158', '15160', '15168', '15170', '15178', '15180', '15188', '15190', '15198', '151a0', '151a8', '151b0', '151b8', '16000', '16008', '16010', '16018', '16020', '16028', '16030', '16038', '16040', '16048', '16050', '16058', '16060', '16068', '16070', '16078', '16080', '16088', '16090', '16098', '160a0', '160a8', '160b0', '160b8', '160c0', '160c8', '160d0', '160d8', '160e0', '160e8', '160f0', '160f8', '16100', '16108', '16110', '16118', '16120', '16128', '16130', '16138', '16140', '16148', '16150', '16158', '16160', '16168', '16170', '16178', '16180', '16188', '16190', '16198', '161a0', '161a8', '161b0', '161b8', '17000', '17008', '17010', '17018', '17020', '17028', '17030', '17038', '17040', '17048', '17050', '17058', '17060', '17068', '17070', '17078', '17080', '17088', '17090', '17098', '170a0', '170a8', '170b0', '170b8', '170c0', '170c8', '170d0', '170d8', '170e0', '170e8', '170f0', '170f8', '17100', '17108', '17110', '17118', '17120', '17128', '17130', '17138', '17140', '17148', '17150', '17158', '17160', '17168', '17170', '17178', '17180', '17188', '17190', '17198', '171a0', '171a8', '171b0', '171b8', '18000', '18008', '18010', '18018', '18020', '18028', '18030', '18038', '18040', '18048', '18050', '18058', '18060', '18068', '18070', '18078', '18080', '18088', '18090', '18098', '180a0', '180a8', '180b0', '180b8', '180c0', '180c8', '180d0', '180d8', '180e0', '180e8', '180f0', '180f8', '18100', '18108', '18110', '18118', '18120', '18128', '18130', '18138', '18140', '18148', '18150', '18158', '18160', '18168', '18170', '18178', '18180', '18188', '18190', '18198', '181a0', '181a8', '181b0', '181b8', '19000', '19008', '19010', '19018', '19020', '19028', '19030', '19038', '19040', '19048', '19050', '19058', '19060', '19068', '19070', '19078', '19080', '19088', '19090', '19098', '190a0', '190a8', '190b0', '190b8', '190c0', '190c8', '190d0', '190d8', '190e0', '190e8', '190f0', '190f8', '19100', '19108', '19110', '19118', '19120', '19128', '19130', '19138', '19140', '19148', '19150', '19158', '19160', '19168', '19170', '19178', '19180', '19188', '19190', '19198', '191a0', '191a8', '191b0', '191b8', '1a000', '1a008', '1a010', '1a018', '1a020', '1a028', '1a030', '1a038', '1a040', '1a048', '1a050', '1a058', '1a060', '1a068', '1a070', '1a078', '1a080', '1a088', '1a090', '1a098', '1a0a0', '1a0a8', '1a0b0', '1a0b8', '1a0c0', '1a0c8', '1a0d0', '1a0d8', '1a0e0', '1a0e8', '1a0f0', '1a0f8', '1a100', '1a108', '1a110', '1a118', '1a120', '1a128', '1a130', '1a138', '1a140', '1a148', '1a150', '1a158', '1a160', '1a168', '1a170', '1a178', '1a180', '1a188', '1a190', '1a198', '1a1a0', '1a1a8', '1a1b0', '1a1b8', '1b000', '1b008', '1b010', '1b018', '1b020', '1b028', '1b030', '1b038', '1b040', '1b048', '1b050', '1b058', '1b060', '1b068', '1b070', '1b078', '1b080', '1b088', '1b090', '1b098', '1b0a0', '1b0a8', '1b0b0', '1b0b8', '1b0c0', '1b0c8', '1b0d0', '1b0d8', '1b0e0', '1b0e8', '1b0f0', '1b0f8', '1b100', '1b108', '1b110', '1b118', '1b120', '1b128', '1b130', '1b138', '1b140', '1b148', '1b150', '1b158', '1b160', '1b168', '1b170', '1b178', '1b180', '1b188', '1b190', '1b198', '1b1a0', '1b1a8', '1b1b0', '1b1b8']


class ActivationFunctions:
    def DeltaFunction(threshold:int):
        return lambda x: int(x >= threshold)
