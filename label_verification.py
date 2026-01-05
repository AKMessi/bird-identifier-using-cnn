import torch
import json

# Load your checkpoint
checkpoint = torch.load("multi_species_model.pth", map_location="cpu")

print("=" * 60)
print("CHECKPOINT ANALYSIS")
print("=" * 60)

# Check what's in the checkpoint
print("\n1. Checkpoint keys:", list(checkpoint.keys()))

# Get the label map
label_map = checkpoint['label_map']
print(f"\n2. Label map type: {type(label_map)}")
print(f"3. Number of classes: {len(label_map)}")

# Show first 10 entries
print("\n4. First 10 entries in label_map:")
for i, (key, value) in enumerate(list(label_map.items())[:10]):
    print(f"   '{key}' -> {value}")

# Check if it's {species: index} or {index: species}
first_key = list(label_map.keys())[0]
first_value = list(label_map.values())[0]
print(f"\n5. First key type: {type(first_key)}, value type: {type(first_value)}")

# Create the correct index -> species mapping
if isinstance(first_key, str):
    print("\n✓ Label map format: {species_name: index}")
    print("   Creating inverse mapping: {index: species_name}")
    
    # Invert the mapping
    index_to_species = {v: k for k, v in label_map.items()}
    
    print("\n6. Correct index -> species mapping (first 10):")
    for i in range(min(10, len(index_to_species))):
        print(f"   Index {i} -> '{index_to_species[i]}'")
    
    # Check for any missing indices
    missing = [i for i in range(len(label_map)) if i not in index_to_species]
    if missing:
        print(f"\n⚠️ WARNING: Missing indices: {missing}")
    else:
        print(f"\n✓ All indices 0-{len(label_map)-1} are mapped correctly")
    
    # Check for duplicate indices
    if len(index_to_species) != len(label_map):
        print(f"\n⚠️ WARNING: Duplicate index values detected!")
        print(f"   Unique indices: {len(index_to_species)}, Total species: {len(label_map)}")
    
else:
    print("\n✓ Label map format: {index: species_name}")
    print("   Already in correct format")
    print("\n6. Index -> species mapping (first 10):")
    for i in range(min(10, len(label_map))):
        print(f"   Index {i} -> '{label_map[i]}'")

# Now load your label_map.json to compare
print("\n" + "=" * 60)
print("COMPARING WITH label_map.json")
print("=" * 60)

try:
    with open('label_map.json', 'r') as f:
        json_label_map = json.load(f)
    
    print(f"\nlabel_map.json entries: {len(json_label_map)}")
    print("First 5 entries:")
    for i, (key, value) in enumerate(list(json_label_map.items())[:5]):
        print(f"   '{key}' -> {value}")
    
    # Check if checkpoint and JSON match
    if label_map == json_label_map:
        print("\n✓ Checkpoint label_map MATCHES label_map.json perfectly!")
    else:
        print("\n⚠️ WARNING: Checkpoint label_map DIFFERS from label_map.json!")
        
        # Find differences
        checkpoint_keys = set(label_map.keys())
        json_keys = set(json_label_map.keys())
        
        only_in_checkpoint = checkpoint_keys - json_keys
        only_in_json = json_keys - checkpoint_keys
        
        if only_in_checkpoint:
            print(f"\n   Species only in checkpoint ({len(only_in_checkpoint)}):")
            for species in list(only_in_checkpoint)[:5]:
                print(f"      {species}")
        
        if only_in_json:
            print(f"\n   Species only in JSON ({len(only_in_json)}):")
            for species in list(only_in_json)[:5]:
                print(f"      {species}")
        
        # Check if indices match for common species
        common_species = checkpoint_keys & json_keys
        mismatched = []
        for species in common_species:
            if label_map[species] != json_label_map[species]:
                mismatched.append(species)
        
        if mismatched:
            print(f"\n   Species with different indices ({len(mismatched)}):")
            for species in mismatched[:5]:
                print(f"      {species}: checkpoint={label_map[species]}, json={json_label_map[species]}")

except FileNotFoundError:
    print("\nlabel_map.json not found - skipping comparison")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

# Create the correct class_names list
if isinstance(first_key, str):
    index_to_species = {v: k for k, v in label_map.items()}
    class_names = [index_to_species[i] for i in range(len(label_map))]
else:
    class_names = [label_map[i] for i in sorted(label_map.keys())]

print("\nTo use in your app, the class_names list should be:")
print(f"class_names = [")
for i in range(min(10, len(class_names))):
    print(f"    # Index {i}")
    print(f"    '{class_names[i]}',")
print(f"    # ... ({len(class_names)} total species)")
print(f"]")

print("\nVerify this by testing with a known bird:")
print("1. Find a recording you KNOW the species of")
print("2. Check what index the model predicts")
print("3. Verify class_names[predicted_index] matches the actual bird")