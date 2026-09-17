#pragma once
// v1 command routing: validate | estimate | run. Returns a SPEC FR-11
// process exit code (0/2/3/4/5/6).

namespace strikecem::cli {

int run(int argc, char** argv);

} // namespace strikecem::cli
