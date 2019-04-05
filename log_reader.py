class LogReader:
    iter_loss_log = {}
    activations_log = {}
    weight_log = {}

    def process_iter_log_line(self, line):
        iter_num = int(line.split()[1].rstrip(','))
        iter_loss = float(line.split()[-1])
        self.iter_loss_log[iter_num] = iter_loss

    def process_activation_log_line(self, line):
        info_line, value_line = line.split('\t')
        unit_type = info_line.split()[2]
        unit_idx = int(info_line.split()[4])

        unit_char = info_line.split()[7].rstrip(')') if "represents char" in info_line else None

        input_char = info_line.split()[-1]
        value = float(value_line)

        if unit_type not in self.activations_log.keys():
            self.activations_log[unit_type] = {}

        if unit_idx not in self.activations_log[unit_type].keys():
            self.activations_log[unit_type][unit_idx] = {}

        self.activations_log[unit_type][unit_idx][input_char] = (
            {
                "unit_char": unit_char,
                "input_char": input_char,
                "value": value
            }
        )

    def process_weight_log_line(self, line):
        info_line, value_line = line.split('\t')
        source_unit_line, target_unit_line = line.split(" to ")

        source_unit_line = source_unit_line.lstrip("weight ")

        source_unit_type = source_unit_line.split()[0]
        source_unit_idx = int(source_unit_line.split()[2])
        source_unit_char = source_unit_line.split()[-1].rstrip(')') if "represents" in source_unit_line else None

        target_unit_type = target_unit_line.split()[0]
        target_unit_idx = target_unit_line.split()[2]
        target_unit_char = target_unit_line.split()[-1].rstrip(')') if "represents" in source_unit_line else None

        value = float(value_line)

        if source_unit_type not in self.weight_log.keys():
            self.weight_log[source_unit_type] = {}

        if source_unit_idx not in self.weight_log[source_unit_type].keys():
            self.weight_log[source_unit_type][source_unit_idx] = {}

        if target_unit_type not in self.weight_log[source_unit_type][source_unit_idx].keys():
            self.weight_log[source_unit_type][source_unit_idx][target_unit_type] = {}

        self.weight_log[source_unit_type][source_unit_idx][target_unit_type][target_unit_idx] = {
            "source_unit_char": source_unit_char,
            "target_unit_char": target_unit_char,
            "value": value
        }

    def parse_log_file(self, log_file):
        for line in log_file:
            line = line.rstrip()
            if line is None:
                continue
            elif line.startswith("iter"):
                self.process_iter_log_line(line)
            elif line.startswith("activation"):
                self.process_activation_log_line(line)
            elif line.startswith("weight"):
                self.process_weight_log_line(line)
