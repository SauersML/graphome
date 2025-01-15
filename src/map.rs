fn parse_paf(paf_path: &str) -> (
    HashMap<String, Vec<AlignmentBlock>>,               // alignment_by_path
    HashMap<String, RefChromAlignments>                 // alignment_by_chr
) {
    let f = File::open(paf_path).expect("Cannot open PAF");
    let reader = BufReader::new(f);

    let mut alignment_by_path = HashMap::new();
    let mut alignment_by_chr  = HashMap::new();

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // typical PAF columns (≥12):
        // qName, qLen, qStart, qEnd, strand, tName, tLen, tStart, tEnd, nMatches, blockLen, mapQ, ...
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 12 {
            continue;
        }
        let q_name  = fields[0].to_string();
        let q_start = fields[2].parse::<usize>().unwrap_or(0);
        let q_end   = fields[3].parse::<usize>().unwrap_or(0);
        let strand_char = fields[4].chars().next().unwrap_or('+');
        let t_name  = fields[5].to_string();
        let t_start = fields[7].parse::<usize>().unwrap_or(0);
        let t_end   = fields[8].parse::<usize>().unwrap_or(0);

        let strand = match strand_char {
            '+' => true,
            _ => false
        };

        let ab = AlignmentBlock {
            path_name: q_name.clone(),
            q_start,
            q_end,
            ref_chr: t_name.clone(),
            r_start: t_start,
            r_end: t_end,
            strand,
        };

        alignment_by_path.entry(q_name).or_insert_with(Vec::new).push(ab.clone());

        // also store in alignment_by_chr
        let refblock = RefBlock {
            r_start: t_start,
            r_end:   t_end,
            ablock:  ab,
        };
        alignment_by_chr.entry(t_name).or_insert_with(RefChromAlignments::new)
                        .push(refblock);
    }

    (alignment_by_path, alignment_by_chr)
}ne
}
