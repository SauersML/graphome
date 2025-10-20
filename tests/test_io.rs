use std::io::{BufRead, BufReader};

use flate2::read::GzDecoder;

use graphome::io;

#[test]
fn test_decode_remote_gfa_header() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.gfa.gz";
    let reader = io::open(url)?;
    let decoder = GzDecoder::new(reader);
    let mut buf_reader = BufReader::new(decoder);
    let mut header = String::new();
    buf_reader.read_line(&mut header)?;
    assert_eq!(header.trim_end(), "H\tVN:Z:1.1\tRS:Z:GRCh38 CHM13");
    Ok(())
}
