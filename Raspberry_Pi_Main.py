'''
0 = E619FF00
1 = BA45FF00
2 = B946FF00
3 = B847FF00
4 = BB44FF00
5 = BF40FF00
6 = BC43FF00
7 = F807FF00
8 = EA15FF00
9 = F609FF00
* = E916FF00
# = F20DFF00
UP = E718FF00
LEFT = F708FF00
RIGHT = A55AFF00
DOWN = AD52FF00
OK = E31CFF00
'''
Of course this is assuming you get one exactly like the one I got. You might want to test yourself. I used IDE 2.3.6 with IRremote 3.5.2 library from Arminjo, shirriff, z3t0.

Here is the code:

#include <IRremote.h>

int IRPIN = 11;

void setup()

{

Serial.begin(9600);

Serial.println("Enabling IRin");

IrReceiver.begin(IRPIN, ENABLE_LED_FEEDBACK);

Serial.println("Enabled IRin");

}

void loop()

{

if (IrReceiver.decode())

{

Serial.println(IrReceiver.decodedIRData.decodedRawData, HEX);

IrReceiver.resume();

}

delay(500);

}
