#include <algorithm>
#include <cstdint>
#include <cstdio>


extern "C" void interpolate_ts_on_seconds_border(unsigned int input_size,
                                                 unsigned int output_size,
                                                 const uint64_t * times,
                                                 const uint64_t * values,
                                                 unsigned int time_step,
                                                 uint64_t * output)
{
    auto output_end = (*times / time_step) * time_step;
    auto output_begin = output_end - time_step;
    auto output_cell = output;

    auto input_cell = values;
    auto input_time = times;
    auto input_val = *input_cell;
    auto input_begin = *input_time - time_step;
    auto input_end = *input_time;
    auto rate = ((double)*input_cell) / (input_end - input_begin);

    // output array mush fully cover input array
    while(output_cell < output + output_size) {
        // check if cells intersect
        auto intersection = ((int64_t)std::min(output_end, input_end)) - std::max(output_begin, input_begin);

        // add intersection slice to output array
        if(intersection > 0) {
            auto slice = (uint64_t)(intersection * rate);
            *output_cell += slice;
            input_val -= slice;
        }

        // switch to next input or output cell
        if (output_end >= input_end){
            *output_cell += input_val;

            ++input_cell;
            ++input_time;

            if(input_time == times + input_size)
                return;

            input_val = *input_cell;
            input_begin = input_end;
            input_end = *input_time;
            rate = ((double)*input_cell) / (input_end - input_begin);
        } else {
            ++output_cell;
            output_begin = output_end;
            output_end += time_step;
        }
    }
}


extern "C" unsigned int interpolate_ts_on_seconds_border_qd(unsigned int input_size,
                                                            unsigned int output_size,
                                                            const uint64_t * times,
                                                            const uint64_t * values,
                                                            unsigned int time_step,
                                                            uint64_t * output)
{
    auto input_end = times + input_size;
    auto curr_output_tm = *times - time_step / 2;

    for(auto output_cell = output; output_cell < output + output_size; ++output_cell) {
        while (curr_output_tm > *times) {
            if (++times >= input_end)
                return output_cell - output;
            ++values;
        }
        *output_cell = *values;
        curr_output_tm += time_step;
    }

    return output_size;
}
