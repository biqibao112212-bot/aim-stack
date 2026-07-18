#pragma once

// Simulator builds intentionally exclude the real robot CAN/serial layer.
// Some copied vivsionn headers include Serial.h transitively but do not use the
// Serial type in the simulator-facing control path.
namespace rm
{
class Serial;
}

