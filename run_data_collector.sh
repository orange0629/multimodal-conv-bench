# Quick test: 500 COCO images                                                                                                                  
  python pipeline/data_collector.py --dataset coco --data-dir data/ --max-images 500                                                               
                                                                                                                                                   
  # Full COCO val (5k images)
  python pipeline/data_collector.py --dataset coco --data-dir data/                                                                                
                                                                                                                                                 
  # Visual Genome subset                                                                                                                           
  python pipeline/data_collector.py --dataset vg --data-dir data/ --max-images 1000