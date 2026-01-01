#!/usr/bin/env python3
"""
Test script to verify HSIFusion Lightning Pro Classifier implementation.

This script validates:
1. Correct inheritance from base model
2. Proper removal of reconstruction branches
3. Classification head functionality
4. Multi-scale feature aggregation
5. Different pooling modes
6. Embedding extraction
7. Integration with the base HSIFusionNet Lightning Pro

Author: Verification Test Suite
Date: 2025-10-21
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Import the classifier module
from hsifusion_classifier_v253 import (
    HSIFusionNetV25LightningProClassifier,
    LightningProClassifierHeadConfig,
    create_hsifusion_lightning_classifier,
)
from hsifusion_v252_complete import (
    HSIFusionNetV25LightningPro,
    LightningProConfig,
)


def test_inheritance():
    """Verify classifier correctly inherits from base model."""
    print("\n" + "="*70)
    print("TEST 1: Inheritance and Architecture")
    print("="*70)

    backbone_config = LightningProConfig(
        in_channels=3,
        out_channels=10,  # Will be overridden by head config
        base_channels=64,
        depths=[2, 2, 4, 2],
        num_heads=4,
    )
    head_config = LightningProClassifierHeadConfig(num_classes=10)

    classifier = HSIFusionNetV25LightningProClassifier(
        backbone_config=backbone_config,
        head_config=head_config,
    )

    # Check inheritance
    assert isinstance(classifier, HSIFusionNetV25LightningPro), \
        "Classifier must inherit from HSIFusionNetV25LightningPro"
    print("[PASS] Correct inheritance from HSIFusionNetV25LightningPro")

    # Check that reconstruction branches are removed
    assert classifier.decoder_stages is None, "decoder_stages should be None"
    assert classifier.upsample_layers is None, "upsample_layers should be None"
    assert classifier.cross_attns is None, "cross_attns should be None"
    assert classifier.output_head is None, "output_head should be None"
    print("[PASS] Reconstruction branches properly removed")

    # Check classifier components exist
    assert hasattr(classifier, 'head'), "Classifier should have 'head' attribute"
    assert hasattr(classifier, 'normalizer'), "Classifier should have 'normalizer'"
    assert hasattr(classifier, 'num_classes'), "Classifier should have 'num_classes'"
    assert hasattr(classifier, 'embedding_dim'), "Classifier should have 'embedding_dim'"
    print("[PASS] Classification components properly initialized")

    # Check encoder is preserved
    assert hasattr(classifier, 'encoder_stages'), "Encoder stages should be preserved"
    assert hasattr(classifier, 'stem'), "Stem should be preserved"
    print("[PASS] Encoder architecture preserved")

    print("\n[OK] TEST 1 PASSED: Inheritance and architecture verified")
    return True


def test_forward_pass():
    """Test basic forward pass with classification output."""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass and Output Shape")
    print("="*70)

    num_classes = 15
    batch_size = 4
    in_channels = 3
    height, width = 128, 128

    # Create classifier
    classifier = create_hsifusion_lightning_classifier(
        model_size='tiny',
        in_channels=in_channels,
        num_classes=num_classes,
        compile_mode=None,  # Disable compilation for testing
    )

    # Create input
    x = torch.randn(batch_size, in_channels, height, width)

    # Test basic forward pass
    with torch.no_grad():
        logits = classifier(x)

    # Check output shape
    expected_shape = (batch_size, num_classes)
    assert logits.shape == expected_shape, \
        f"Expected output shape {expected_shape}, got {logits.shape}"
    print(f"[PASS] Output shape correct: {logits.shape}")

    # Check output is valid (no NaN or Inf)
    assert not torch.isnan(logits).any(), "Output contains NaN values"
    assert not torch.isinf(logits).any(), "Output contains Inf values"
    print("[PASS] Output values are valid (no NaN/Inf)")

    print("\n[OK] TEST 2 PASSED: Forward pass produces correct output")
    return True


def test_decoder_error():
    """Verify that calling forward_decoder raises an error."""
    print("\n" + "="*70)
    print("TEST 3: Decoder Removal Verification")
    print("="*70)

    classifier = create_hsifusion_lightning_classifier(
        model_size='tiny',
        num_classes=10,
        compile_mode=None,
    )

    # Create dummy input
    x = torch.randn(2, 128, 16, 16)
    encoder_features = [torch.randn(2, 128, 16, 16) for _ in range(4)]

    # Try to call forward_decoder - should raise RuntimeError
    try:
        classifier.forward_decoder(x, encoder_features)
        assert False, "forward_decoder should raise RuntimeError"
    except RuntimeError as e:
        assert "removes the reconstruction decoder" in str(e), \
            f"Unexpected error message: {e}"
        print(f"[PASS] forward_decoder correctly raises error: {e}")

    print("\n[OK] TEST 3 PASSED: Decoder properly disabled")
    return True


def test_pooling_modes():
    """Test different pooling modes (avg, max, avgmax)."""
    print("\n" + "="*70)
    print("TEST 4: Pooling Modes")
    print("="*70)

    pooling_modes = ['avg', 'max', 'avgmax']
    num_classes = 10
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)

    for pooling in pooling_modes:
        head_config = LightningProClassifierHeadConfig(
            num_classes=num_classes,
            pooling=pooling,
        )
        backbone_config = LightningProConfig(
            in_channels=3,
            base_channels=64,
            depths=[2, 2, 2, 2],
            num_heads=4,
        )

        classifier = HSIFusionNetV25LightningProClassifier(
            backbone_config=backbone_config,
            head_config=head_config,
        )

        with torch.no_grad():
            logits = classifier(x)

        assert logits.shape == (batch_size, num_classes), \
            f"Pooling '{pooling}' failed: expected shape {(batch_size, num_classes)}, got {logits.shape}"
        print(f"[PASS] Pooling mode '{pooling}' works correctly")

    print("\n[OK] TEST 4 PASSED: All pooling modes functional")
    return True


def test_multi_scale_features():
    """Test multi-scale feature aggregation."""
    print("\n" + "="*70)
    print("TEST 5: Multi-Scale Feature Aggregation")
    print("="*70)

    num_classes = 10
    x = torch.randn(2, 3, 64, 64)

    # Test with multi-scale enabled
    head_config_multi = LightningProClassifierHeadConfig(
        num_classes=num_classes,
        use_multi_scale=True,
    )
    backbone_config = LightningProConfig(
        in_channels=3,
        base_channels=64,
        depths=[2, 2, 2, 2],
        num_heads=4,
    )

    classifier_multi = HSIFusionNetV25LightningProClassifier(
        backbone_config=backbone_config,
        head_config=head_config_multi,
    )

    # Test with single scale
    head_config_single = LightningProClassifierHeadConfig(
        num_classes=num_classes,
        use_multi_scale=False,
    )

    classifier_single = HSIFusionNetV25LightningProClassifier(
        backbone_config=backbone_config,
        head_config=head_config_single,
    )

    with torch.no_grad():
        logits_multi = classifier_multi(x)
        logits_single = classifier_single(x)

    # Both should produce valid outputs
    assert logits_multi.shape == (2, num_classes)
    assert logits_single.shape == (2, num_classes)

    # Multi-scale should use more features
    multi_scales = len(classifier_multi.selected_scales)
    single_scales = len(classifier_single.selected_scales)

    print(f"[PASS] Multi-scale uses {multi_scales} scales")
    print(f"[PASS] Single-scale uses {single_scales} scale(s)")
    assert multi_scales > single_scales, "Multi-scale should use more scales"

    # Embedding dimension should be different
    assert classifier_multi.embedding_dim > classifier_single.embedding_dim, \
        "Multi-scale should have larger embedding dimension"
    print(f"[PASS] Multi-scale embedding dim: {classifier_multi.embedding_dim}")
    print(f"[PASS] Single-scale embedding dim: {classifier_single.embedding_dim}")

    print("\n[OK] TEST 5 PASSED: Multi-scale aggregation working")
    return True


def test_embedding_extraction():
    """Test embedding extraction functionality."""
    print("\n" + "="*70)
    print("TEST 6: Embedding Extraction")
    print("="*70)

    num_classes = 10
    batch_size = 3
    x = torch.randn(batch_size, 3, 64, 64)

    classifier = create_hsifusion_lightning_classifier(
        model_size='tiny',
        num_classes=num_classes,
        compile_mode=None,
    )

    # Test extract_embeddings method
    with torch.no_grad():
        embeddings = classifier.extract_embeddings(x)

    assert embeddings.shape == (batch_size, classifier.embedding_dim), \
        f"Expected embedding shape {(batch_size, classifier.embedding_dim)}, got {embeddings.shape}"
    print(f"[PASS] Embedding extraction works: shape {embeddings.shape}")

    # Test forward with return_embeddings=True
    with torch.no_grad():
        logits, embeddings2 = classifier(x, return_embeddings=True)

    assert logits.shape == (batch_size, num_classes)
    assert embeddings2.shape == (batch_size, classifier.embedding_dim)
    print(f"[PASS] Forward with return_embeddings=True works")

    # Embeddings should be the same
    assert torch.allclose(embeddings, embeddings2, atol=1e-5), \
        "Embeddings from different methods should match"
    print(f"[PASS] Embeddings consistent across methods")

    # Test return_selected_features
    with torch.no_grad():
        logits, embeddings, features = classifier(
            x,
            return_embeddings=True,
            return_selected_features=True
        )

    assert isinstance(features, list), "Features should be a list"
    assert len(features) == len(classifier.selected_scales), \
        f"Should return {len(classifier.selected_scales)} feature maps"
    print(f"[PASS] Feature extraction returns {len(features)} feature maps")

    print("\n[OK] TEST 6 PASSED: Embedding extraction functional")
    return True


def test_hidden_layer():
    """Test classifier with hidden layer."""
    print("\n" + "="*70)
    print("TEST 7: Hidden Layer Configuration")
    print("="*70)

    num_classes = 10
    hidden_dim = 256
    x = torch.randn(2, 3, 64, 64)

    # Test with hidden layer
    head_config_with_hidden = LightningProClassifierHeadConfig(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=0.1,
    )
    backbone_config = LightningProConfig(
        in_channels=3,
        base_channels=64,
        depths=[2, 2, 2, 2],
        num_heads=4,
    )

    classifier_with_hidden = HSIFusionNetV25LightningProClassifier(
        backbone_config=backbone_config,
        head_config=head_config_with_hidden,
    )

    # Test without hidden layer
    head_config_no_hidden = LightningProClassifierHeadConfig(
        num_classes=num_classes,
        hidden_dim=None,
    )

    classifier_no_hidden = HSIFusionNetV25LightningProClassifier(
        backbone_config=backbone_config,
        head_config=head_config_no_hidden,
    )

    # Both should work
    with torch.no_grad():
        logits_with = classifier_with_hidden(x)
        logits_no = classifier_no_hidden(x)

    assert logits_with.shape == (2, num_classes)
    assert logits_no.shape == (2, num_classes)

    # Check internal structure
    assert isinstance(classifier_with_hidden.head, nn.Sequential), \
        "Classifier with hidden layer should use Sequential"
    assert isinstance(classifier_no_hidden.head, nn.Linear), \
        "Classifier without hidden layer should use Linear"

    print(f"[PASS] Classifier with hidden layer ({hidden_dim}) works")
    print(f"[PASS] Classifier without hidden layer works")

    print("\n[OK] TEST 7 PASSED: Hidden layer configuration working")
    return True


def test_model_sizes():
    """Test different model sizes."""
    print("\n" + "="*70)
    print("TEST 8: Model Size Variants")
    print("="*70)

    model_sizes = ['tiny', 'small', 'base', 'large', 'xlarge']
    num_classes = 10
    x = torch.randn(1, 3, 64, 64)

    for size in model_sizes:
        try:
            classifier = create_hsifusion_lightning_classifier(
                model_size=size,
                num_classes=num_classes,
                compile_mode=None,
            )

            with torch.no_grad():
                logits = classifier(x)

            assert logits.shape == (1, num_classes)

            # Count parameters
            params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
            print(f"[PASS] Model size '{size}' works: {params/1e6:.2f}M parameters")

        except Exception as e:
            print(f"[FAIL] Model size '{size}' failed: {e}")
            return False

    print("\n[OK] TEST 8 PASSED: All model sizes functional")
    return True


def test_selected_scales():
    """Test custom scale selection."""
    print("\n" + "="*70)
    print("TEST 9: Custom Scale Selection")
    print("="*70)

    num_classes = 10
    x = torch.randn(2, 3, 64, 64)

    # Test selecting specific scales
    head_config = LightningProClassifierHeadConfig(
        num_classes=num_classes,
        selected_scales=[1, 3],  # Use only scales 1 and 3
    )
    backbone_config = LightningProConfig(
        in_channels=3,
        base_channels=64,
        depths=[2, 2, 2, 2],
        num_heads=4,
    )

    classifier = HSIFusionNetV25LightningProClassifier(
        backbone_config=backbone_config,
        head_config=head_config,
    )

    with torch.no_grad():
        logits = classifier(x)

    assert logits.shape == (2, num_classes)
    assert classifier.selected_scales == (1, 3), \
        f"Expected selected_scales (1, 3), got {classifier.selected_scales}"

    print(f"[PASS] Custom scale selection works: {classifier.selected_scales}")

    print("\n[OK] TEST 9 PASSED: Custom scale selection working")
    return True


def test_classification_workflow():
    """Test a complete classification workflow."""
    print("\n" + "="*70)
    print("TEST 10: Complete Classification Workflow")
    print("="*70)

    # Simulate a small classification task
    num_classes = 5
    num_samples = 10
    batch_size = 4

    # Create classifier
    classifier = create_hsifusion_lightning_classifier(
        model_size='tiny',
        num_classes=num_classes,
        compile_mode=None,
        classifier_head={
            'pooling': 'avg',
            'use_multi_scale': True,
            'dropout': 0.1,
        }
    )

    print(f"[PASS] Created classifier with {num_classes} classes")

    # Create fake data
    images = [torch.randn(3, 64, 64) for _ in range(num_samples)]
    labels = torch.randint(0, num_classes, (num_samples,))

    print(f"[PASS] Created {num_samples} samples")

    # Test training mode
    classifier.train()

    # Mini-batch
    batch_images = torch.stack(images[:batch_size])
    batch_labels = labels[:batch_size]

    # Forward pass
    logits = classifier(batch_images)
    assert logits.shape == (batch_size, num_classes)
    print(f"[PASS] Forward pass in training mode: {logits.shape}")

    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, batch_labels)
    assert loss.item() > 0
    print(f"[PASS] Loss computation works: {loss.item():.4f}")

    # Test eval mode
    classifier.eval()
    with torch.no_grad():
        logits_eval = classifier(batch_images)
        predictions = logits_eval.argmax(dim=1)

    assert predictions.shape == (batch_size,)
    print(f"[PASS] Predictions in eval mode: {predictions}")

    # Test embeddings for similarity
    with torch.no_grad():
        emb1 = classifier.extract_embeddings(batch_images[:2])
        emb2 = classifier.extract_embeddings(batch_images[:2])

    # Same inputs should give same embeddings
    assert torch.allclose(emb1, emb2, atol=1e-5)
    print(f"[PASS] Embedding extraction deterministic")

    print("\n[OK] TEST 10 PASSED: Complete workflow functional")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*80)
    print(" HSIFusion Lightning Pro Classifier - Verification Test Suite")
    print("="*80)

    tests = [
        ("Inheritance and Architecture", test_inheritance),
        ("Forward Pass and Output Shape", test_forward_pass),
        ("Decoder Removal", test_decoder_error),
        ("Pooling Modes", test_pooling_modes),
        ("Multi-Scale Feature Aggregation", test_multi_scale_features),
        ("Embedding Extraction", test_embedding_extraction),
        ("Hidden Layer Configuration", test_hidden_layer),
        ("Model Size Variants", test_model_sizes),
        ("Custom Scale Selection", test_selected_scales),
        ("Complete Classification Workflow", test_classification_workflow),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] TEST FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")

    print("\n" + "-"*80)
    print(f"Results: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*80)

    if passed == total:
        print("\n*** ALL TESTS PASSED - Classifier implementation is VERIFIED! ***")
        print("\nThe HSIFusion Lightning Pro Classifier:")
        print("  [OK] Correctly inherits from the base model")
        print("  [OK] Properly removes reconstruction branches")
        print("  [OK] Provides classification functionality")
        print("  [OK] Supports multi-scale feature aggregation")
        print("  [OK] Supports different pooling modes (avg, max, avgmax)")
        print("  [OK] Can extract embeddings for downstream tasks")
        print("  [OK] Works with all model sizes (tiny to xlarge)")
        print("  [OK] Supports custom configurations")
        print("\n[OK] CONCLUSION: The module is correctly implemented and production-ready!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed - please review errors above")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
